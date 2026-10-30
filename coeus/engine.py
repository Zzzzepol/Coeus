import asyncio
import hashlib
from collections.abc import Callable
from urllib.parse import urlsplit

from coeus.config import Settings
from coeus.extraction import extract_chunk
from coeus.models import Critique, Document, Evidence, Plan, Report, Source
from coeus.providers import ProviderError, RequestTooLarge
from coeus.report import render_report, report_claims, validate_report
from coeus.sources import authority_hint, diverse_evidence, ranked_results
from coeus.storage import Store, now


def fingerprint(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:16]


class ResearchEngine:
    def __init__(
        self,
        settings: Settings,
        store: Store,
        provider,
        progress: Callable[[str], None] | None = None,
    ):
        self.settings, self.store, self.provider = settings, store, provider
        self.progress = progress or (lambda message: None)

    async def run(
        self,
        question: str,
        documents: list[Document] | None = None,
        mode: str = "live",
        resume_from: str | None = None,
        on_started=None,
    ) -> str:
        if resume_from:
            previous = self.store.get(resume_from)
            question, mode = previous["question"], previous["mode"]
            documents = [
                Document(name=s["title"], text=s["content"])
                for s in previous["artifacts"].get("source", {}).values()
                if s["kind"] == "document"
            ]
        question = question.strip()
        if not 5 <= len(question) <= 2000:
            raise ValueError("Enter a research question between 5 and 2,000 characters.")
        documents = documents or []
        if len(documents) > 5:
            raise ValueError("Upload at most five documents.")
        if mode not in {"live", "offline", "hybrid"}:
            raise ValueError("Unknown run mode.")
        if mode == "offline" and not documents:
            raise ValueError("Offline research requires at least one text or Markdown document.")
        run_id = (
            self.store.clone_run(resume_from)
            if resume_from
            else self.store.create_run(question, mode)
        )
        if on_started:
            on_started(run_id)

        def emit(message):
            self.store.event(run_id, message)
            self.progress(message)

        self.provider.on_event = emit
        try:
            async with asyncio.timeout(self.settings.run_timeout):
                await self._research(run_id, question, documents, mode, emit)
        except asyncio.CancelledError:
            self.store.finish(run_id, "cancelled", error="Research was cancelled.")
            raise
        except Exception as exc:  # noqa: BLE001 -- run boundary must persist unexpected failures
            # Only controlled errors go into the UI/database; provider errors never contain secrets.

            message = (
                str(exc)
                if isinstance(exc, ProviderError)
                else (
                    "Run timed out."
                    if isinstance(exc, TimeoutError)
                    else f"Research could not complete ({type(exc).__name__}). Inspect saved events and artifacts."
                )
            )
            self.store.finish(run_id, "failed", error=message)
            emit(message)
        return run_id

    async def _research(self, run_id, question, documents, mode, emit):
        cfg = self.settings
        saved = self.store.get(run_id)["artifacts"]
        self.store.put(
            run_id,
            "config",
            "settings",
            {
                "model": cfg.llm_model if mode == "live" else cfg.ollama_model,
                "workers": cfg.workers if mode == "live" else 1,
                "max_tasks": cfg.max_tasks,
                "profile": cfg.profile,
                "preferred_domains": cfg.preferred_domains,
                "results_per_task": cfg.results_per_task,
                "run_timeout": cfg.run_timeout,
                "extract_chunk_chars": cfg.extract_chunk_chars,
                "evidence_budget_bytes": cfg.evidence_budget_bytes,
                "llm_max_output_tokens": cfg.llm_max_output_tokens,
            },
        )
        plan = None
        if mode != "offline":
            if saved.get("plan", {}).get("plan", {}).get("tasks"):
                plan = Plan.model_validate(saved["plan"]["plan"])
                emit("Reusing the saved research plan")
            else:
                emit("Planning research tasks")
                plan = await self.provider.generate(
                    "plan",
                    f"Create 1–{cfg.max_tasks} focused search tasks. Prioritize official documentation, "
                    "original research and primary evidence. Include preferred_domains with likely "
                    "primary-source hostnames when known; these are ranking hints, not verified authority. "
                    "Include conflicting evidence. "
                    "For time-sensitive claims seek dated sources; separate products from models.",
                    {"question": question, "documents": [d.name for d in documents]},
                    Plan,
                )
                plan.tasks = plan.tasks[: cfg.max_tasks]
                self.store.put(run_id, "plan", "plan", plan)
        else:
            emit("Planning document research (offline; web search disabled)")
        sources = {
            key: Source.model_validate(value) for key, value in saved.get("source", {}).items()
        }
        evidence = [Evidence.model_validate(value) for value in saved.get("evidence", {}).values()]
        warnings: list[str] = []
        document_sources = []
        for index, doc in enumerate(documents):
            source = Source(
                id="D" + fingerprint(f"{index}:{doc.name}:{doc.text}"),
                title=doc.name,
                uri="document:" + doc.name,
                kind="document",
                content=doc.text,
            )
            source = next(
                (
                    existing
                    for existing in sources.values()
                    if existing.kind == "document"
                    and existing.title == doc.name
                    and existing.content == doc.text
                ),
                source,
            )
            sources[source.id] = source
            self.store.put(run_id, "source", source.id, source)
            # Extract across the entire document, in bounded chunks rather than silently truncating it.
            for offset in range(0, len(doc.text), cfg.extract_chunk_chars):
                document_sources.append(
                    (source, doc.text[offset : offset + cfg.extract_chunk_chars])
                )

        specs = saved.get("task_spec", {})
        if not specs:
            if plan:
                specs = {
                    f"T{i}": {"kind": "web", **task.model_dump()}
                    for i, task in enumerate(plan.tasks, 1)
                }
            for i, (source, chunk) in enumerate(document_sources, 1):
                # Persist the exact excerpt to keep retries independent of future chunk settings.
                specs[f"D{i}"] = {
                    "kind": "document",
                    "source_id": source.id,
                    "content": chunk,
                    "objective": source.title,
                }
            for key, spec in specs.items():
                self.store.put(run_id, "task_spec", key, spec)
        if mode == "offline":
            self.store.put(
                run_id,
                "plan",
                "plan",
                {
                    "scope": "Uploaded documents only",
                    "tasks": [
                        {"id": key, "objective": spec["objective"]} for key, spec in specs.items()
                    ],
                },
            )
        queue = asyncio.Queue()
        for key, spec in specs.items():
            previous_task = saved.get("task", {}).get(key, {})
            if (
                previous_task.get("status") == "complete"
                and previous_task.get("findings", 0) > 0
                and not previous_task.get("rejected", 0)
            ):
                emit(f"Reusing completed task {key}")
                continue
            self.store.put(
                run_id,
                "task",
                key,
                {"kind": spec["kind"], "status": "queued", "objective": spec["objective"]},
            )
            queue.put_nowait((key, spec["kind"], spec))

        async def worker():
            while True:
                try:
                    task_id, kind, task = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                task_info = {
                    "kind": kind,
                    "status": "running",
                    "objective": task["objective"],
                }
                self.store.put(run_id, "task", task_id, task_info)
                try:
                    emit(f"Researching {task_id}: {task_info['objective']}")
                    batch = []
                    source_gap = False
                    if kind == "web":
                        preferred = cfg.preferred_domains + task.get("preferred_domains", [])
                        cached = saved.get("search", {}).get(task_id, {}).get("source_ids", [])
                        if cached:
                            candidates = [
                                {
                                    "url": sources[key].uri,
                                    "title": sources[key].title,
                                    "content": sources[key].content,
                                    "published_date": sources[key].published_date,
                                }
                                for key in cached
                                if key in sources
                            ]
                            emit(f"Reusing saved sources for {task_id}")
                        else:
                            candidates = await self.provider.search(task["query"])
                        primary_search = getattr(self.provider, "search_primary", None)
                        has_preferred = any(
                            isinstance(r, dict)
                            and isinstance(r.get("url"), str)
                            and authority_hint(r["url"], preferred) == "preferred domain"
                            for r in candidates
                        )
                        if preferred and primary_search and not has_preferred:
                            emit(f"Checking preferred source domains for {task_id}")
                            try:
                                candidates = (
                                    await primary_search(task["query"], preferred) + candidates
                                )
                            except ProviderError as exc:
                                warnings.append(
                                    f"{task_id}: preferred-source search failed ({exc}); general results used."
                                )
                        candidates = [
                            r
                            for r in candidates
                            if isinstance(r, dict)
                            and isinstance(r.get("raw_content") or r.get("content"), str)
                            and (r.get("raw_content") or r.get("content")).strip()
                        ]
                        results = ranked_results(candidates, cfg.results_per_task, preferred)
                        for result in results:
                            uri = result["url"]
                            content = result.get("raw_content") or result.get("content")
                            date = result.get("published_date")
                            source = Source(
                                id="S" + fingerprint(uri + content[:12_000]),
                                title=str(result.get("title") or uri)[:500],
                                uri=uri,
                                kind="web",
                                content=content[:12_000],
                                domain=(urlsplit(uri).hostname or "").removeprefix("www."),
                                retrieved_at=now(),
                                published_date=str(date)[:80] if date else None,
                                authority_hint=authority_hint(uri, preferred),
                            )
                            if source.id in sources:
                                source.retrieved_at = sources[source.id].retrieved_at
                            sources[source.id] = source
                            self.store.put(run_id, "source", source.id, source)
                            batch.append(source.model_dump())
                        source_gap = bool(preferred) and not any(
                            s["authority_hint"] == "preferred domain" for s in batch
                        )
                        if source_gap:
                            warnings.append(
                                f"{task_id}: no source from the requested preferred domains was found."
                            )
                        self.store.put(
                            run_id, "search", task_id, {"source_ids": [s["id"] for s in batch]}
                        )
                    else:
                        source = sources[task["source_id"]]
                        batch = [{**source.model_dump(), "content": task["content"]}]
                    if not batch:
                        raise ValueError("No usable sources")
                    accepted = rejected = failed_chunks = 0
                    for source_data in batch:
                        text = source_data["content"]
                        for offset in range(0, len(text), cfg.extract_chunk_chars):
                            chunk = text[offset : offset + cfg.extract_chunk_chars]
                            checkpoint = fingerprint(
                                task_id + source_data["id"] + str(offset) + chunk
                            )
                            if (
                                saved.get("chunk", {}).get(checkpoint, {}).get("status")
                                == "complete"
                            ):
                                continue
                            chunk_rejected = chunk_failed = 0
                            # Metadata is bounded separately; identifiers and content remain exact.
                            excerpt = {
                                **source_data,
                                "content": chunk,
                                "title": source_data["title"][:200],
                                "uri": source_data["uri"][:500],
                            }
                            async for findings, error in extract_chunk(
                                self.provider, question, excerpt
                            ):
                                if error:
                                    failed_chunks += 1
                                    chunk_failed += 1
                                    warnings.append(f"{task_id}: {error}")
                                    continue
                                for f in findings.findings:
                                    if f.source_id != source_data["id"] or f.quote not in chunk:
                                        rejected += 1
                                        chunk_rejected += 1
                                        continue
                                    item = Evidence(
                                        **f.model_dump(),
                                        task_id=task_id,
                                        id="E"
                                        + fingerprint(task_id + f.source_id + f.quote + f.claim),
                                    )
                                    if not any(e.id == item.id for e in evidence):
                                        evidence.append(item)
                                        self.store.put(run_id, "evidence", item.id, item)
                                        accepted += 1
                            self.store.put(
                                run_id,
                                "chunk",
                                checkpoint,
                                {
                                    "task_id": task_id,
                                    "source_id": source_data["id"],
                                    "status": "partial"
                                    if chunk_rejected or chunk_failed
                                    else "complete",
                                },
                            )
                    accepted = sum(e.task_id == task_id for e in evidence)
                    task_info.update(
                        status="partial" if failed_chunks or source_gap else "complete",
                        source_gap=source_gap,
                        findings=accepted,
                        rejected=rejected,
                        failed_chunks=failed_chunks,
                    )
                    if rejected:
                        warnings.append(f"{task_id}: rejected {rejected} unverifiable findings.")
                    if accepted == 0:
                        warnings.append(f"{task_id}: no verified relevant findings.")
                except asyncio.CancelledError:
                    task_info.update(status="cancelled")
                    raise
                except Exception as exc:  # noqa: BLE001 -- isolate failed jobs from other workers
                    reason = str(exc) if isinstance(exc, ProviderError) else type(exc).__name__
                    task_info.update(status="failed", error=reason)
                    warnings.append(f"{task_id}: research failed ({reason}).")
                    emit(f"Task {task_id} failed; continuing with other evidence")
                finally:
                    self.store.put(run_id, "task", task_id, task_info)
                    queue.task_done()
                    emit(
                        f"Task {task_id}: {task_info['status']} · {task_info.get('findings', 0)} verified findings"
                    )

        async with asyncio.TaskGroup() as group:
            for _ in range(cfg.workers if mode == "live" else 1):
                group.create_task(worker())
        self.store.put(run_id, "warnings", "warnings", warnings)
        tasks = list(self.store.get(run_id)["artifacts"].get("task", {}).values())
        coverage = {
            "total": len(tasks),
            "completed": sum(t["status"] == "complete" for t in tasks),
            "with_evidence": sum(t.get("findings", 0) > 0 for t in tasks),
            "failed": sum(t["status"] in {"failed", "partial"} for t in tasks),
            "rejected": sum(t.get("rejected", 0) for t in tasks),
        }
        coverage["status"] = (
            "complete"
            if (
                coverage["completed"] == coverage["total"]
                and coverage["with_evidence"] == coverage["total"]
                and not coverage["rejected"]
            )
            else "partial"
        )
        self.store.put(run_id, "coverage", "research", coverage)
        if not evidence:
            emit("No verified evidence was collected; no report will be fabricated")
            raise ValueError("No verified evidence")
        # Explicit context budget; all evidence remains inspectable even when not synthesized.
        evidence.sort(key=lambda e: (e.task_id, e.id))
        selected, used_chars = [], 0
        for e in diverse_evidence(evidence, sources):
            size = len(e.model_dump_json().encode("utf-8"))
            if used_chars + size <= cfg.evidence_budget_bytes:
                selected.append(e)
                used_chars += size
        coverage["evidence_used"] = len(selected)
        coverage["evidence_collected"] = len(evidence)
        if len(selected) < len(evidence):
            coverage["status"] = "partial"
            warnings.append(
                f"Report context uses {len(selected)} of {len(evidence)} findings; "
                "all findings are retained in SQLite."
            )
        self.store.put(run_id, "coverage", "research", coverage)
        coverage["sources"] = len(sources)
        coverage["domains"] = len(
            {s.domain or urlsplit(s.uri).hostname for s in sources.values() if s.kind == "web"}
        )
        self.store.put(run_id, "coverage", "research", coverage)
        data = {
            "question": question,
            "evidence": [e.model_dump() for e in selected],
            "coverage": coverage,
            "source_metadata": [
                {
                    "id": s.id,
                    "uri": s.uri[:500],
                    "published_date": s.published_date,
                    "authority_hint": s.authority_hint,
                }
                for s in sources.values()
                if s.id in {e.source_id for e in selected}
            ],
            "scope": "Uploaded documents only"
            if mode == "offline"
            else "Web and uploaded documents",
        }
        emit("Writing a cited draft")
        for attempt in range(3):
            try:
                draft = await self.provider.generate(
                    "draft",
                    "Answer using only the evidence. Every claim needs valid evidence_ids. "
                    "Provide a concise direct answer in answer, with evidence_ids, then practical tradeoffs in "
                    "at most six claims. Prefer primary evidence. Include limitations and distinguish source statements "
                    "from established conclusions. Do not repeat the answer verbatim in findings.",
                    data,
                    Report,
                )
                break
            except RequestTooLarge:
                if attempt == 2 or len(selected) <= 1:
                    raise
                selected = selected[: max(1, len(selected) // 2)]
                data["evidence"] = [e.model_dump() for e in selected]
                coverage.update(status="partial", evidence_used=len(selected))
                self.store.put(run_id, "coverage", "research", coverage)
                warnings.append(
                    "Report evidence context was reduced after a provider size rejection."
                )
        validate_report(draft, selected)
        self.store.put(run_id, "report", "draft", draft)

        def review_data(report):
            cited = {ref for c in report_claims(report) for ref in c.evidence_ids}
            return {**data, "evidence": [e.model_dump() for e in selected if e.id in cited]}

        def claim_map(report):
            return {
                **({"answer": report.answer.model_dump()} if report.answer else {}),
                **{f"claim_{i}": c.model_dump() for i, c in enumerate(report.claims, 1)},
            }

        async def critique(report):
            return await self.provider.generate(
                "critic",
                "Audit each claim against its exact evidence quotes. Flag unsupported inference, "
                "contradictions, missing scope, and misleading certainty. Verdict must be revise "
                "if any claim is not supported. Return a claim_reviews entry for EVERY supplied "
                "claim_id, including answer. Mark supported=false for contradictions or claims "
                "whose cited quotes do not establish them. Suggest concrete repairs.",
                {**review_data(report), "report": report.model_dump(), "claims": claim_map(report)},
                Critique,
            )

        emit("Critic checking claims against evidence")
        review = await critique(draft)
        self.store.put(run_id, "critique", "initial", review)

        def rejected_claims(report, review):
            mapped = {}
            duplicate = set()
            for item in review.claim_reviews:
                if item.claim_id in mapped:
                    duplicate.add(item.claim_id)
                mapped[item.claim_id] = item
            return {
                key: (
                    "Critic returned duplicate assessments for this claim."
                    if key in duplicate
                    else mapped[key].reason
                    if key in mapped
                    else "No explicit support assessment was returned."
                )
                for key in claim_map(report)
                if key in duplicate or key not in mapped or not mapped[key].supported
            }

        if review.verdict == "revise" or review.issues or rejected_claims(draft, review):
            emit("Revising the report from critic feedback")
            draft = await self.provider.generate(
                "revise",
                "Repair only unsupported claims identified in claim_reviews. Preserve supported claims "
                "and their citations unchanged. Rewrite or remove unsupported claims. "
                "Use only the supplied evidence and preserve explicit limitations.",
                {
                    **review_data(draft),
                    "draft": draft.model_dump(),
                    "critique": review.model_dump(),
                },
                Report,
            )
            validate_report(draft, selected)
            review = await critique(draft)
            self.store.put(run_id, "critique", "final", review)
        rejected = rejected_claims(draft, review)
        excluded = [
            {"claim_id": key, "text": claim_map(draft)[key]["text"], "reason": reason}
            for key, reason in rejected.items()
        ]
        self.store.put(run_id, "excluded", "claims", excluded)
        if excluded:
            draft = draft.model_copy(
                update={
                    "answer": None if "answer" in rejected else draft.answer,
                    "claims": [
                        c for i, c in enumerate(draft.claims, 1) if f"claim_{i}" not in rejected
                    ],
                    "limitations": (
                        [
                            f"{len(excluded)} claims were removed because their evidence support was unresolved."
                        ]
                        + draft.limitations
                    )[:12],
                }
            )
        review_status = (
            "passed"
            if review.verdict == "pass"
            and not review.issues
            and not excluded
            and report_claims(draft)
            else "needs review"
        )
        if review_status != "passed":
            warnings += review.issues or ["The report did not receive complete per-claim approval."]
        self.store.put(run_id, "report", "final", draft)
        self.store.put(run_id, "warnings", "warnings", warnings)
        markdown = render_report(
            question,
            draft,
            selected,
            list(sources.values()),
            mode,
            review_status,
            warnings,
            coverage,
            excluded,
        )
        self.store.finish(
            run_id,
            (
                "needs_review"
                if review_status != "passed"
                else "partial"
                if coverage["status"] == "partial"
                else "complete"
            ),
            markdown,
        )
        emit(f"Report saved — critic {review_status}")
