import asyncio
import hashlib
from collections.abc import Callable
from urllib.parse import urlsplit

from coeus.config import Settings
from coeus.extraction import extract_chunk
from coeus.models import Critique, Document, Evidence, Plan, Report, Source
from coeus.providers import ProviderError, RequestTooLarge
from coeus.report import render_report, report_claims, validate_report
from coeus.storage import Store


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
        self, question: str, documents: list[Document] | None = None, mode: str = "live"
    ) -> str:
        question = question.strip()
        if not 5 <= len(question) <= 2000:
            raise ValueError("Enter a research question between 5 and 2,000 characters.")
        documents = documents or []
        if len(documents) > 5:
            raise ValueError("Upload at most five documents.")
        if mode not in {"live", "offline"}:
            raise ValueError("Unknown run mode.")
        if mode == "offline" and not documents:
            raise ValueError("Offline research requires at least one text or Markdown document.")
        run_id = self.store.create_run(question, mode)

        def emit(message):
            self.store.event(run_id, message)
            self.progress(message)

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
        self.store.put(
            run_id,
            "config",
            "settings",
            {
                "model": cfg.llm_model if mode == "live" else cfg.ollama_model,
                "workers": cfg.workers if mode == "live" else 1,
                "max_tasks": cfg.max_tasks,
                "results_per_task": cfg.results_per_task,
                "run_timeout": cfg.run_timeout,
                "extract_chunk_chars": cfg.extract_chunk_chars,
                "evidence_budget_bytes": cfg.evidence_budget_bytes,
                "llm_max_output_tokens": cfg.llm_max_output_tokens,
            },
        )
        plan = None
        if mode == "live":
            emit("Planning research tasks")
            plan = await self.provider.generate(
                "plan",
                f"Create 1–{cfg.max_tasks} focused web search tasks to answer the question. "
                "Seek primary evidence and conflicting findings. Document names are context only.",
                {"question": question, "documents": [d.name for d in documents]},
                Plan,
            )
            plan.tasks = plan.tasks[: cfg.max_tasks]
            self.store.put(run_id, "plan", "plan", plan)
        else:
            emit("Planning document research (offline; web search disabled)")
        sources: dict[str, Source] = {}
        evidence: list[Evidence] = []
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
            sources[source.id] = source
            self.store.put(run_id, "source", source.id, source)
            # Extract across the entire document, in bounded chunks rather than silently truncating it.
            for offset in range(0, len(doc.text), cfg.extract_chunk_chars):
                document_sources.append(
                    (source, doc.text[offset : offset + cfg.extract_chunk_chars])
                )

        queue = asyncio.Queue()
        if plan:
            for i, task in enumerate(plan.tasks, 1):
                queue.put_nowait((f"T{i}", "web", task))
        for i, chunk in enumerate(document_sources, 1):
            queue.put_nowait((f"D{i}", "document", chunk))

        if mode == "offline":
            self.store.put(
                run_id,
                "plan",
                "plan",
                {
                    "scope": "Uploaded documents only; no web search",
                    "tasks": [
                        {
                            "id": f"D{i}",
                            "document": source.title,
                            "objective": "Extract evidence relevant to the research question",
                        }
                        for i, (source, _) in enumerate(document_sources, 1)
                    ],
                },
            )

        async def worker():
            while True:
                try:
                    task_id, kind, task = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                task_info = {
                    "kind": kind,
                    "status": "running",
                    "objective": task.objective if kind == "web" else task[0].title,
                }
                self.store.put(run_id, "task", task_id, task_info)
                try:
                    emit(f"Researching {task_id}: {task_info['objective']}")
                    batch = []
                    if kind == "web":
                        for result in await self.provider.search(task.query):
                            uri = result.get("url", "")
                            content = result.get("raw_content") or result.get("content") or ""
                            if not isinstance(uri, str) or not isinstance(content, str):
                                continue
                            if urlsplit(uri).scheme not in {"https", "http"} or not content.strip():
                                continue
                            # Snapshot ID includes content so quote provenance survives changed search results.
                            source = Source(
                                id="S" + fingerprint(uri + content[:12_000]),
                                title=str(result.get("title") or uri),
                                uri=uri,
                                kind="web",
                                content=content[:12_000],
                            )
                            sources[source.id] = source
                            self.store.put(run_id, "source", source.id, source)
                            batch.append(source.model_dump())
                    else:
                        source, chunk = task
                        batch = [{**source.model_dump(), "content": chunk}]
                    if not batch:
                        raise ValueError("No usable sources")
                    accepted = rejected = failed_chunks = 0
                    for source_data in batch:
                        text = source_data["content"]
                        for offset in range(0, len(text), cfg.extract_chunk_chars):
                            chunk = text[offset : offset + cfg.extract_chunk_chars]
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
                                    warnings.append(f"{task_id}: {error}")
                                    continue
                                for f in findings.findings:
                                    if f.source_id != source_data["id"] or f.quote not in chunk:
                                        rejected += 1
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
                    task_info.update(
                        status="partial" if failed_chunks else "complete",
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
        for e in evidence:
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
        data = {
            "question": question,
            "evidence": [e.model_dump() for e in selected],
            "coverage": coverage,
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
                    "claims. Prefer primary evidence. Include limitations and distinguish source statements "
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

        async def critique(report):
            return await self.provider.generate(
                "critic",
                "Audit each claim against its exact evidence quotes. Flag unsupported inference, "
                "contradictions, missing scope, and misleading certainty. Verdict must be revise "
                "if any claim is not supported. Suggest concrete repairs.",
                {**review_data(report), "report": report.model_dump()},
                Critique,
            )

        emit("Critic checking claims against evidence")
        review = await critique(draft)
        self.store.put(run_id, "critique", "initial", review)
        if review.verdict == "revise" or review.issues:
            emit("Revising the report from critic feedback")
            draft = await self.provider.generate(
                "revise",
                "Repair the draft using the critique. Remove unsupported claims. "
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
        review_status = (
            "passed" if review.verdict == "pass" and not review.issues else "needs review"
        )
        if review_status != "passed":
            warnings += review.issues or ["The critic requested further revision."]
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
