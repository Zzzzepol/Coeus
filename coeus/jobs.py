"""In-process background jobs for the personal Streamlit UI; no external queue."""

import asyncio
import threading
from pathlib import Path

from coeus import providers
from coeus.engine import ResearchEngine
from coeus.storage import Store

_active = {}
_registry_lock = threading.Lock()


def active_job(db_path):
    with _registry_lock:
        job = _active.get(str(Path(db_path).resolve()))
        return job if job and not job.done.is_set() else None


class Job:
    def __init__(self, settings, question, documents, mode, resume_from):
        self.settings = settings.model_copy(deep=True)
        self.question, self.documents, self.mode = question, documents, mode
        self.resume_from = resume_from
        self.run_id = None
        self.error = None
        self.cancelled = False
        self.done = threading.Event()
        self.cancel_requested = threading.Event()
        self.lock = threading.Lock()
        self.loop = None
        self.task = None

    def snapshot(self):
        with self.lock:
            return {
                "run_id": self.run_id,
                "error": self.error,
                "done": self.done.is_set(),
                "cancelled": self.cancelled,
            }

    def cancel(self):
        self.cancel_requested.set()
        with self.lock:
            if self.loop and self.task and not self.loop.is_closed():
                try:
                    self.loop.call_soon_threadsafe(self.task.cancel)
                except RuntimeError:
                    pass  # The run may have closed its loop just before this cancellation.

    def _started(self, run_id):
        with self.lock:
            self.run_id = run_id

    async def _execute(self):
        with self.lock:
            self.loop = asyncio.get_running_loop()
            self.task = asyncio.current_task()
        provider = providers.create_provider(self.settings, self.mode)
        try:
            if self.cancel_requested.is_set():
                raise asyncio.CancelledError()
            await ResearchEngine(self.settings, Store(self.settings.db_path), provider).run(
                self.question,
                self.documents,
                self.mode,
                resume_from=self.resume_from,
                on_started=self._started,
            )
        finally:
            await provider.close()

    def _run(self):
        try:
            asyncio.run(self._execute())
        except asyncio.CancelledError:
            self.cancelled = True
        except Exception as exc:  # noqa: BLE001 -- worker boundary must communicate failure to UI
            self.error = (
                str(exc)
                if isinstance(exc, providers.ProviderError)
                else f"Cannot start research ({type(exc).__name__}); check mode, documents and configuration."
            )
        finally:
            self.done.set()
            with _registry_lock:
                key = str(Path(self.settings.db_path).resolve())
                if _active.get(key) is self:
                    _active.pop(key)


def start_job(settings, question, documents=None, mode="live", resume_from=None):
    if resume_from:
        parent = Store(settings.db_path).get(resume_from)
        question, mode = parent["question"], parent["mode"]
    job = Job(settings, question, documents or [], mode, resume_from)
    key = str(Path(settings.db_path).resolve())
    with _registry_lock:
        current = _active.get(key)
        if current and not current.done.is_set():
            raise ValueError("A research job is already active in this workspace.")
        _active[key] = job
    threading.Thread(target=job._run, name="coeus-research", daemon=True).start()
    return job
