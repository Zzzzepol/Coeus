import asyncio
import hashlib
from collections.abc import Callable
from urllib.parse import urlsplit

from coeus.config import Settings
from coeus.models import Critique, Document, Evidence, Findings, Plan, Report, Source
from coeus.providers import ProviderError
from coeus.report import render_report, validate_report
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
        if mode not in {"live", "demo"}:
            raise ValueError("Unknown run mode.")
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
                "model": cfg.llm_model if mode == "live" else "offline-fixtures",
                "workers": cfg.workers,
                "max_tasks": cfg.max_tasks,
                "results_per_task": cfg.results_per_task,
                "run_timeout": cfg.run_timeout,
            },
        )
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
            for offset in range(0, len(doc.text), 10_000):
                document_sources.append((source, doc.text[offset : offset + 10_000]))

        queue = asyncio.Queue()
        for i, task in enumerate(plan.tasks, 1):
            queue.put_nowait((f"T{i}", "web", task))
        for i, chunk in enumerate(document_sources, 1):
            queue.put_nowait((f"D{i}", "document", chunk))

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
                    findings = await self.provider.generate(
                        "extract",
                        "Extract up to 6 relevant findings. Each needs an exact, contiguous quote "
                        "copied from source content and its source_id. Return no findings if irrelevant.",
                        {"question": question, "sources": batch},
                        Findings,
                    )
                    allowed = {s["id"]: s["content"] for s in batch}
                    accepted = 0
                    for f in findings.findings:
                        if f.source_id not in allowed or f.quote not in allowed[f.source_id]:
                            warnings.append(
                                f"{task_id}: rejected a finding with an unverifiable quote."
                            )
                            continue
                        item = Evidence(
                            **f.model_dump(),
                            task_id=task_id,
                            id="E" + fingerprint(task_id + f.source_id + f.quote + f.claim),
                        )
                        if not any(e.id == item.id for e in evidence):
                            evidence.append(item)
                            self.store.put(run_id, "evidence", item.id, item)
                            accepted += 1
                    task_info.update(status="complete", findings=accepted)
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
            for _ in range(cfg.workers):
                group.create_task(worker())
        self.store.put(run_id, "warnings", "warnings", warnings)
        if not evidence:
            emit("No verified evidence was collected; no report will be fabricated")
            raise ValueError("No verified evidence")
        # Explicit context budget; all evidence remains inspectable even when not synthesized.
        evidence.sort(key=lambda e: (e.task_id, e.id))
        selected, used_chars = [], 0
        for e in evidence:
            size = len(e.model_dump_json())
            if used_chars + size <= 48_000:
                selected.append(e)
                used_chars += size
        if len(selected) < len(evidence):
            warnings.append(
                f"Report context uses {len(selected)} of {len(evidence)} findings; "
                "all findings are retained in SQLite."
            )
        data = {
            "question": question,
            "evidence": [e.model_dump() for e in selected],
            "research_warnings": warnings,
        }
        emit("Writing a cited draft")
        draft = await self.provider.generate(
            "draft",
            "Answer using only the evidence. Every claim needs valid evidence_ids. "
            "Include limitations and distinguish source statements from established conclusions.",
            data,
            Report,
        )
        validate_report(draft, selected)
        self.store.put(run_id, "report", "draft", draft)

        async def critique(report):
            return await self.provider.generate(
                "critic",
                "Audit each claim against its exact evidence quotes. Flag unsupported inference, "
                "contradictions, missing scope, and misleading certainty. Verdict must be revise "
                "if any claim is not supported. Suggest concrete repairs.",
                {**data, "report": report.model_dump()},
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
                {**data, "draft": draft.model_dump(), "critique": review.model_dump()},
                Report,
            )
            validate_report(draft, selected)
            review = await critique(draft)
            self.store.put(run_id, "critique", "final", review)
        review_status = (
            "passed" if review.verdict == "pass" and not review.issues else "needs review"
        )
        warnings += [
            (
                "Web evidence may be search excerpts or truncated page content; "
                "citations verify provenance, not factual truth."
            )
        ]
        if review_status != "passed":
            warnings += review.issues or ["The critic requested further revision."]
        self.store.put(run_id, "report", "final", draft)
        self.store.put(run_id, "warnings", "warnings", warnings)
        markdown = render_report(
            question, draft, selected, list(sources.values()), mode, review_status, warnings
        )
        self.store.finish(
            run_id, "complete" if review_status == "passed" else "needs_review", markdown
        )
        emit(f"Report saved — critic {review_status}")
