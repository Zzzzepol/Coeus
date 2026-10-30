import asyncio
import json
import threading
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime

import httpx

from coeus.config import Settings
from coeus.engine import ResearchEngine
from coeus.jobs import active_job, start_job
from coeus.models import Claim, ClaimReview, Critique, Report
from coeus.providers import HostedProvider, HybridProvider, ProviderError
from coeus.scheduling import retry_delay
from coeus.sources import ranked_results
from coeus.storage import Store
from tests.fakes import FixtureProvider


def workspace(tmp_path, **kw):
    settings = Settings(
        _env_file=None, COEUS_DB_PATH=tmp_path / "research.db", COEUS_REQUEST_INTERVAL=0, **kw
    )
    return settings, Store(settings.db_path)


def test_retry_only_failed_task_preserves_parent(tmp_path):
    class Partial(FixtureProvider):
        async def search(self, query):
            if "bounded" in query:
                raise ProviderError("temporary failure")
            return await super().search(query)

    class Retry(FixtureProvider):
        def __init__(self):
            self.searches = []

        async def search(self, query):
            self.searches.append(query)
            return await super().search(query)

        async def generate(self, operation, *args):
            assert operation != "plan"
            return await super().generate(operation, *args)

    cfg, store = workspace(tmp_path)
    parent_id = asyncio.run(ResearchEngine(cfg, store, Partial()).run("How is evidence saved?"))
    parent = store.get(parent_id)
    assert parent["status"] == "partial"
    retry = Retry()
    child_id = asyncio.run(ResearchEngine(cfg, store, retry).run("", resume_from=parent_id))
    child = store.get(child_id)
    assert child_id != parent_id
    assert child["status"] == "complete"
    assert retry.searches == ["bounded research workers"]
    assert store.get(parent_id) == parent
    assert child["artifacts"]["attempt"]["parent"]["run_id"] == parent_id
    assert set(parent["artifacts"]["evidence"]) <= set(child["artifacts"]["evidence"])


def test_retry_reuses_sources_and_successful_excerpt(tmp_path):
    class First(FixtureProvider):
        async def search(self, query):
            return [
                {
                    "url": "https://example.org/source",
                    "title": "Source",
                    "content": "A" * 1000 + "B" * 1000,
                }
            ]

        async def generate(self, operation, instructions, data, schema):
            if operation == "extract" and data["sources"][0]["content"].startswith("B"):
                raise ProviderError("temporary failure")
            return await super().generate(operation, instructions, data, schema)

    class Retry(FixtureProvider):
        async def search(self, query):
            raise AssertionError("Saved sources should be reused")

        async def generate(self, operation, instructions, data, schema):
            assert operation != "plan"
            if operation == "extract":
                assert data["sources"][0]["content"] == "B" * 1000
            return await super().generate(operation, instructions, data, schema)

    cfg, store = workspace(tmp_path, COEUS_MAX_TASKS=1, COEUS_EXTRACT_CHUNK_CHARS=1000)
    parent = asyncio.run(ResearchEngine(cfg, store, First()).run("What does the source establish?"))
    child = asyncio.run(ResearchEngine(cfg, store, Retry()).run("", resume_from=parent))
    result = store.get(child)
    assert result["status"] == "complete"
    assert len(result["artifacts"]["evidence"]) == 2
    assert len(result["artifacts"]["source"]) == 1


def test_unsupported_answer_and_claim_are_removed(tmp_path):
    class Contradictory(FixtureProvider):
        async def generate(self, operation, instructions, data, schema):
            if schema is Report:
                ref = data["evidence"][0]["id"]
                return Report(
                    title="Evidence review",
                    answer=Claim(text="Unsupported direct answer", evidence_ids=[ref]),
                    claims=[
                        Claim(text="Supported finding", evidence_ids=[ref]),
                        Claim(text="Unsupported finding", evidence_ids=[ref]),
                    ],
                    limitations=["Limited sources"],
                )
            if schema is Critique:
                return Critique(
                    verdict="revise",
                    issues=["Two claims contradict evidence"],
                    suggestions=[],
                    claim_reviews=[
                        ClaimReview(
                            claim_id=key,
                            supported=key == "claim_1",
                            reason="Supported"
                            if key == "claim_1"
                            else "Contradicts the quoted evidence",
                        )
                        for key in data["claims"]
                    ],
                )
            return await super().generate(operation, instructions, data, schema)

    cfg, store = workspace(tmp_path)
    run_id = asyncio.run(ResearchEngine(cfg, store, Contradictory()).run("What is established?"))
    result = store.get(run_id)
    final = result["artifacts"]["report"]["final"]
    assert final["answer"] is None
    assert [c["text"] for c in final["claims"]] == ["Supported finding"]
    assert len(result["artifacts"]["excluded"]["claims"]) == 2
    main, limitations = result["report"].split("## Limitations")
    assert "Unsupported" not in main
    assert "Unsupported finding" in limitations
    assert result["status"] == "needs_review"


def test_global_pass_without_claim_assessments_is_not_accepted(tmp_path):
    class Unmapped(FixtureProvider):
        async def generate(self, operation, instructions, data, schema):
            if schema is Critique:
                return Critique(verdict="pass", issues=[], suggestions=[])
            return await super().generate(operation, instructions, data, schema)

    cfg, store = workspace(tmp_path)
    run_id = asyncio.run(ResearchEngine(cfg, store, Unmapped()).run("What is established?"))
    result = store.get(run_id)
    assert result["status"] == "needs_review"
    assert result["artifacts"]["report"]["final"]["claims"] == []
    assert "No findings passed" in result["report"]


def test_source_ranking_deduplicates_and_spreads_domains():
    results = [
        {"url": "https://blog.example/story?utm_source=a", "title": "Blog"},
        {"url": "https://docs.example.org/a", "title": "Docs A", "published_date": "2026-01-02"},
        {"url": "https://docs.example.org/b", "title": "Docs B"},
        {"url": "https://blog.example/story?utm_source=b#intro", "title": "Duplicate"},
        {"url": "javascript:alert(1)", "title": "Invalid"},
    ]
    selected = ranked_results(results, 2, ["example.org"])
    assert [r["title"] for r in selected] == ["Docs A", "Blog"]
    assert selected[0]["published_date"] == "2026-01-02"
    assert selected[1]["url"] == "https://blog.example/story"


def test_rate_limit_pauses_other_workers(monkeypatch):
    real_sleep = asyncio.sleep

    async def check():
        cooling, release = asyncio.Event(), asyncio.Event()
        calls = []

        async def sleep(delay):
            if delay == 5:
                cooling.set()
                await release.wait()
            else:
                await real_sleep(0)

        monkeypatch.setattr("coeus.scheduling.asyncio.sleep", sleep)
        provider = HostedProvider(
            Settings(
                _env_file=None, LLM_API_KEY="test", TAVILY_API_KEY="test", COEUS_REQUEST_INTERVAL=0
            )
        )
        await provider.client.aclose()

        def handler(request):
            calls.append(request)
            return (
                httpx.Response(429, headers={"retry-after": "5"})
                if len(calls) == 1
                else httpx.Response(200, json={"results": []})
            )

        provider.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            first = asyncio.create_task(provider.search("first"))
            await cooling.wait()
            second = asyncio.create_task(provider.search("second"))
            await real_sleep(0)
            assert len(calls) == 1
            release.set()
            await asyncio.gather(first, second)
            assert len(calls) == 3
        finally:
            await provider.close()

    asyncio.run(check())


def test_retry_after_date_and_long_delays_are_not_cut_short():
    future = format_datetime(datetime.now(UTC) + timedelta(seconds=90))
    assert 88 <= retry_delay(future, 0) <= 90
    assert retry_delay("120", 0) == 120
    assert retry_delay("nan", 0) == 5


def test_hybrid_sends_only_search_to_tavily():
    async def check():
        provider = HybridProvider(
            Settings(
                _env_file=None,
                TAVILY_API_KEY="search-only",
                LLM_API_KEY="",
                COEUS_REQUEST_INTERVAL=0,
            )
        )
        await provider.client.aclose()
        hosts = []

        def handler(request):
            hosts.append(request.url.host)
            if request.url.host == "api.tavily.com":
                assert request.headers["authorization"] == "Bearer search-only"
                assert json.loads(request.content)["query"] == "question"
                return httpx.Response(200, json={"results": []})
            assert request.url.host == "127.0.0.1"
            assert "authorization" not in request.headers
            if request.url.path == "/api/show":
                return httpx.Response(200, json={})
            return httpx.Response(
                200,
                json={
                    "message": {
                        "content": '{"verdict":"pass","issues":[],"suggestions":[],"claim_reviews":[]}'
                    }
                },
            )

        provider.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            await provider.search("question")
            await provider.generate("critic", "Review", {}, Critique)
            assert set(hosts) == {"api.tavily.com", "127.0.0.1"}
        finally:
            await provider.close()

    asyncio.run(check())


def test_background_job_cancel_preserves_state_and_closes_provider(tmp_path, monkeypatch):
    started = threading.Event()

    class Slow(FixtureProvider):
        closed = False

        async def generate(self, *args):
            started.set()
            await asyncio.sleep(60)

        async def close(self):
            self.closed = True

    provider = Slow()
    monkeypatch.setattr("coeus.providers.create_provider", lambda cfg, mode: provider)
    cfg, store = workspace(tmp_path)
    job = start_job(cfg, "What is established?")
    try:
        assert started.wait(3)
        assert active_job(cfg.db_path) is job
        job.cancel()
        assert job.done.wait(3)
        assert store.get(job.run_id)["status"] == "cancelled"
        assert provider.closed
        assert active_job(cfg.db_path) is None
    finally:
        job.cancel()


def test_profiles_set_distinct_bounded_research_budgets():
    cfg = Settings(_env_file=None)
    quick, thorough = cfg.with_profile("quick"), cfg.with_profile("thorough")
    assert quick.max_tasks * quick.results_per_task == 4
    assert thorough.max_tasks * thorough.results_per_task == 20
    assert quick.run_timeout < thorough.run_timeout
    assert cfg.profile == "custom"


def test_missing_primary_source_triggers_targeted_search(tmp_path):
    class Targeted(FixtureProvider):
        targeted = 0

        async def search_primary(self, query, domains):
            assert domains == ["sqlite.org"]
            self.targeted += 1
            return [
                {
                    "url": "https://www.sqlite.org/whentouse.html",
                    "title": "Official source",
                    "content": "Only one writer can write to a SQLite database file at a time.",
                    "published_date": "2026-01-01",
                }
            ]

    cfg, store = workspace(tmp_path, COEUS_MAX_TASKS=1, COEUS_RESULTS_PER_TASK=1)
    cfg.preferred_domains = ["sqlite.org"]
    provider = Targeted()
    run_id = asyncio.run(ResearchEngine(cfg, store, provider).run("What does SQLite support?"))
    result = store.get(run_id)
    assert provider.targeted == 1
    assert result["status"] == "complete"
    source = next(iter(result["artifacts"]["source"].values()))
    assert source["domain"] == "sqlite.org"
    assert source["authority_hint"] == "preferred domain"
    assert source["published_date"] == "2026-01-01"
    assert "Published: 2026-01-01" in result["report"]


def test_missing_preferred_source_is_partial_even_if_claims_pass(tmp_path):
    cfg, store = workspace(tmp_path, COEUS_MAX_TASKS=1)
    cfg.preferred_domains = ["sqlite.org"]
    run_id = asyncio.run(
        ResearchEngine(cfg, store, FixtureProvider()).run("What does SQLite support?")
    )
    result = store.get(run_id)
    assert result["status"] == "partial"
    assert result["artifacts"]["task"]["T1"]["source_gap"]
    assert "Claim review:** passed" in result["report"]


def test_ui_can_cancel_a_background_run(tmp_path, monkeypatch):
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    started = threading.Event()

    class Slow(FixtureProvider):
        async def generate(self, *args):
            started.set()
            await asyncio.sleep(60)

    cfg, store = workspace(tmp_path)
    monkeypatch.setenv("COEUS_DB_PATH", str(cfg.db_path))
    monkeypatch.setattr("coeus.providers.create_provider", lambda cfg, mode: Slow())
    app = AppTest.from_file(
        Path(__file__).resolve().parents[1] / "app.py", default_timeout=15
    ).run()
    app.radio[0].set_value("live").run()
    app.text_area[0].set_value("What do the sources establish?")
    app.button[0].click().run()
    job = active_job(cfg.db_path)
    try:
        assert started.wait(3)
        assert job is not None
        next(b for b in app.button if b.label == "Cancel research").click().run()
        assert job.done.wait(3)
        app.run()
        assert not app.exception
        assert store.get(job.run_id)["status"] == "cancelled"
    finally:
        if job:
            job.cancel()


def test_preferred_domains_match_www_aliases():
    from coeus.sources import authority_hint

    assert authority_hint("https://sqlite.org/faq.html", ["www.sqlite.org"]) == "preferred domain"
    assert authority_hint("https://www.sqlite.org/faq.html", ["sqlite.org"]) == "preferred domain"
