import asyncio
from pathlib import Path

import httpx
import pytest
from streamlit.testing.v1 import AppTest

from coeus.config import Settings
from coeus.engine import ResearchEngine
from coeus.models import Claim, Critique, Document, Finding, Findings, Report
from coeus.providers import DemoProvider, HostedProvider, ProviderError
from coeus.report import plain, validate_report
from coeus.storage import Store


def setup(tmp_path, **overrides):
    cfg = Settings(_env_file=None, COEUS_DB_PATH=tmp_path / "research.sqlite3", **overrides)
    return cfg, Store(cfg.db_path)


def run(tmp_path, provider=None, docs=None, **overrides):
    cfg, store = setup(tmp_path, **overrides)
    run_id = asyncio.run(
        ResearchEngine(cfg, store, provider or DemoProvider()).run(
            "How is evidence stored?", docs, mode="demo"
        )
    )
    return store.get(run_id)


def test_complete_run_persists_and_cites(tmp_path):
    result = run(
        tmp_path,
        docs=[Document(name="notes.md", text="Local documents preserve original evidence.")],
    )
    assert result["status"] == "complete"
    assert "DEMO" in result["report"]
    assert "[^E" in result["report"]
    assert "uploaded document" in result["report"]
    assert set(result["artifacts"]) >= {"source", "evidence", "plan", "critique", "report", "task"}
    assert Store(tmp_path / "research.sqlite3").get(result["id"]) == result
    for e in result["artifacts"]["evidence"].values():
        assert e["quote"] in result["artifacts"]["source"][e["source_id"]]["content"]


def test_concurrency_is_bounded_and_documents_fully_chunked(tmp_path):
    class Tracking(DemoProvider):
        active = 0
        peak = 0

        def __init__(self):
            self.chunks = []

        async def generate(self, operation, instructions, data, schema):
            if operation == "extract":
                self.active += 1
                self.peak = max(self.peak, self.active)
                self.chunks.extend(s["content"] for s in data["sources"] if s["kind"] == "document")
                try:
                    await asyncio.sleep(0.01)
                    return await super().generate(operation, instructions, data, schema)
                finally:
                    self.active -= 1
            return await super().generate(operation, instructions, data, schema)

    provider = Tracking()
    text = "Evidence from a document. " * 1000
    result = run(tmp_path, provider, [Document(name="long.md", text=text)], COEUS_WORKERS=2)
    assert result["status"] == "complete"
    assert provider.peak == 2
    assert "".join(provider.chunks) == text.strip()


def test_partial_search_failure_still_produces_report(tmp_path):
    class Partial(DemoProvider):
        async def search(self, query):
            if "bounded" in query:
                raise ProviderError("Provider returned HTTP 503.")
            return await super().search(query)

    result = run(tmp_path, Partial())
    assert result["status"] == "complete"
    assert "research failed" in result["report"]
    assert result["artifacts"]["task"]["T1"]["status"] == "failed"


def test_fabricated_quotes_fail_without_report(tmp_path):
    class Fabricated(DemoProvider):
        async def generate(self, operation, instructions, data, schema):
            if schema is Findings:
                return Findings(
                    findings=[
                        Finding(
                            source_id=data["sources"][0]["id"],
                            quote="This invented quote does not exist in the source.",
                            claim="Unsupported",
                        )
                    ]
                )
            return await super().generate(operation, instructions, data, schema)

    result = run(tmp_path, Fabricated())
    assert result["status"] == "failed"
    assert result["report"] is None
    assert "evidence" not in result["artifacts"]


def test_critic_revision_is_bounded_and_visible(tmp_path):
    class Skeptic(DemoProvider):
        revisions = 0

        async def generate(self, operation, instructions, data, schema):
            if schema is Critique:
                return Critique(
                    verdict="revise", issues=["Scope remains uncertain"], suggestions=[]
                )
            if operation == "revise":
                self.revisions += 1
            return await super().generate(operation, instructions, data, schema)

    provider = Skeptic()
    result = run(tmp_path, provider)
    assert provider.revisions == 1
    assert result["status"] == "needs_review"
    assert "Scope remains uncertain" in result["report"]
    assert "final" in result["artifacts"]["critique"]


def test_unknown_citation_is_rejected():
    report = Report(
        title="Invalid citation",
        claims=[Claim(text="A claim", evidence_ids=["fake"])],
        limitations=["Unknown evidence"],
    )
    with pytest.raises(ValueError, match="unknown evidence"):
        validate_report(report, [])
    assert plain("[fake](https://example.com)<script>").startswith(r"\[fake\]")


def test_deadline_saves_failure(tmp_path):
    class Slow(DemoProvider):
        async def search(self, query):
            await asyncio.sleep(5)

    result = run(tmp_path, Slow(), COEUS_RUN_TIMEOUT=1)
    assert result["status"] == "failed"
    assert result["error"] == "Run timed out."


def test_cancellation_is_saved(tmp_path):
    cfg, store = setup(tmp_path)

    class Cancelled(DemoProvider):
        async def generate(self, *args):
            raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(ResearchEngine(cfg, store, Cancelled()).run("A valid question"))
    assert store.runs()[0]["status"] == "cancelled"


def test_http_contract_and_invalid_output_retry(tmp_path):
    async def check():
        cfg = Settings(_env_file=None, LLM_API_KEY="secret", TAVILY_API_KEY="secret")
        provider = HostedProvider(cfg)
        await provider.client.aclose()
        attempts = []

        def handler(request):
            attempts.append(request)
            if request.url.host == "api.tavily.com":
                assert request.headers["authorization"] == "Bearer secret"
                return httpx.Response(200, json={"results": []})
            text = (
                "bad json"
                if len(attempts) == 2
                else '{"verdict":"pass","issues":[],"suggestions":[]}'
            )
            return httpx.Response(200, json={"choices": [{"message": {"content": text}}]})

        provider.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            assert await provider.search("question") == []
            result = await provider.generate("critic", "Review", {}, Critique)
            assert result.verdict == "pass"
            assert len(attempts) == 3
        finally:
            await provider.close()

    asyncio.run(check())


def test_streamlit_demo_flow(tmp_path, monkeypatch):
    monkeypatch.setenv("COEUS_DB_PATH", str(tmp_path / "ui.sqlite3"))
    app = AppTest.from_file(
        Path(__file__).resolve().parents[1] / "app.py", default_timeout=15
    ).run()
    assert not app.exception
    app.text_area[0].set_value("How does evidence persistence work?")
    app.button[0].click().run()
    assert not app.exception
    assert any("DEMO" in item.value for item in app.markdown)
    app.run()
    assert not app.exception
    assert len(app.selectbox[0].options) == 1


def test_credentials_and_endpoint_validation():
    with pytest.raises(ValueError, match="API_KEY"):
        Settings(_env_file=None, LLM_API_KEY="", TAVILY_API_KEY="").validate_live()
    with pytest.raises(ValueError, match="HTTPS"):
        Settings(
            _env_file=None,
            LLM_API_KEY="x",
            TAVILY_API_KEY="x",
            LLM_BASE_URL="http://localhost.example.com/v1",
        ).validate_live()


def test_http_auth_errors_do_not_leak_credentials():
    async def check():
        cfg = Settings(_env_file=None, LLM_API_KEY="private-key", TAVILY_API_KEY="private-key")
        provider = HostedProvider(cfg)
        await provider.client.aclose()
        count = 0

        def handler(request):
            nonlocal count
            count += 1
            return httpx.Response(401, json={"error": "private-key is invalid"})

        provider.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            with pytest.raises(ProviderError, match="HTTP 401") as error:
                await provider.search("test")
            assert "private-key" not in str(error.value)
            assert count == 1
        finally:
            await provider.close()

    asyncio.run(check())


def test_rate_limit_respects_retry_after(monkeypatch):
    delays = []

    async def sleep(delay):
        delays.append(delay)

    monkeypatch.setattr("coeus.providers.asyncio.sleep", sleep)

    async def check():
        provider = HostedProvider(
            Settings(_env_file=None, LLM_API_KEY="test", TAVILY_API_KEY="test")
        )
        await provider.client.aclose()
        responses = iter(
            [
                httpx.Response(429, headers={"retry-after": "7"}),
                httpx.Response(200, json={"results": []}),
            ]
        )
        provider.client = httpx.AsyncClient(
            transport=httpx.MockTransport(lambda req: next(responses))
        )
        try:
            assert await provider.search("test") == []
            assert delays == [7]
        finally:
            await provider.close()

    asyncio.run(check())
