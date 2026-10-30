# Coeus

A standalone local research MVP: question → plan → bounded async web/document research → SQLite evidence → cited Markdown → critic → optional revision.

## Quick start

Requires Python 3.11+.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
cp .env.example .env
streamlit run app.py --server.address 127.0.0.1
```

The UI defaults to an **offline demonstration** with explicitly synthetic sources. It exercises the whole pipeline without credentials. Switch to live research after setting `TAVILY_API_KEY` and `LLM_API_KEY` in `.env`. Groq is the default; configure `LLM_BASE_URL` and `LLM_MODEL` for another OpenAI-compatible chat-completions provider supporting JSON object output. Model availability depends on the provider/account.

One complete run from the terminal:

```sh
coeus 'How does Coeus preserve inspectable evidence?' --demo --document examples/notes.md --output reports/demo.md
# Live (requires both API keys):
coeus 'What are the tradeoffs of SQLite for local research tools?' --output reports/research.md
```

Open Streamlit to inspect previous runs, tasks, source snapshots, quotations, draft/final reports, critic feedback, and event logs. Markdown downloads include source links and exact evidence quotes. Data persists in `data/coeus.sqlite3`; no hosted database or job service is required.

## Scope and guarantees

- Pydantic validates plans, evidence, critic output, and reports. One structured-output retry is allowed.
- `asyncio.Queue` and a fixed worker pool bound combined search/extraction concurrency. Default: 3 workers, 4 web tasks, 3 results per task. Task failures do not discard successful evidence.
- Up to 5 UTF-8 `.txt`/`.md` documents, each at most 100,000 characters / 400 KB. Extraction processes one source at a time in chunks of at most 4,000 characters by default; web snapshots are capped at 12,000 characters. Documents are read as data, never executed.
- Exact quote matching rejects invented quotations. Every report claim must reference known evidence. Citations are rendered in code, rather than trusting model-generated links. These checks establish provenance, **not factual accuracy or entailment**.
- Research coverage is recorded separately from claim review. Failed or partial tasks, tasks without verified findings, rejected quotes, or omitted evidence produce partial coverage even when claim review passes. Coverage describes the planned tasks, not exhaustive knowledge of the topic.
- Reports include a cited direct answer when provided by the writer, key findings, plain-language limitations, numbered evidence footnotes, and a diagnostics appendix. Stable evidence IDs remain in SQLite. Older saved reports retain their original format.
- The critic examines claims against quotes; at most one revision and a second critic pass run. Unresolved concerns produce `needs_review`, visible in the UI/report. A critic failure fails the run and preserves the draft.
- Sources, tasks, evidence, drafts, feedback and events persist incrementally. Empty evidence causes a failed run, never an uncited fabricated report.
- HTTP calls have timeouts and bounded retries for transient errors. The total run deadline defaults to 600 seconds. A report uses at most 8,000 UTF-8 bytes of serialized evidence context by default; any omission is disclosed. All evidence remains in SQLite.
- The UI executes one foreground run per session. Killing the process preserves artifacts but may leave a `running` status; resume, multi-user scheduling and distributed infrastructure are deferred. Do not expose this unauthenticated local UI publicly.
- This MVP does not read PDF/DOCX, crawl arbitrary URLs itself, perform recursive research, or export office documents.

## Configuration

See `.env.example`. Keys are read server-side and are not stored in run artifacts. Live mode sends research queries to Tavily and question/document/evidence text to the configured hosted LLM. SQLite and downloaded reports contain your research data in plaintext. Demo mode makes no network calls.

For a public GitHub repository, keep credentials only in your local `.env`; `.env.example` contains empty key placeholders and is intended to be committed. Local environment files, Streamlit secrets, SQLite databases, `data/`, `reports/`, and `uploads/` are ignored by Git. Keep private input documents outside the repository or inside `uploads/`. Before committing, review `git diff --cached` to check exactly what will be published.

The default maximum is 4 web jobs plus 125 document-chunk jobs. Each web page snapshot is processed in excerpts; each extraction call returns at most 6 findings. HTTP 413 triggers at most two levels of lossless excerpt splitting (at most 7 calls per initial excerpt). Successful excerpts survive a failed sibling. Draft requests rejected with 413 can halve the selected evidence up to twice, with omissions disclosed. Review calls include only cited evidence. Provider limits can still reject the minimum request; such failures remain visible. The default output allowance is 2,048 tokens per LLM call. Adjust `COEUS_EXTRACT_CHUNK_CHARS`, `COEUS_EVIDENCE_BUDGET_BYTES`, and `COEUS_LLM_MAX_OUTPUT_TOKENS` for your provider. Smaller excerpts can increase the number of calls. A successful run makes planner, extraction, writer and critic calls, with up to two additional calls for revision/review. Retries and optional document jobs increase provider usage; this is a bounded run, not a monetary budget guarantee.

## Development

```sh
python -m pytest
ruff check .
```

Tests use deterministic providers and HTTP mocks; they do not spend API credits. The demo verifies orchestration and storage, not live provider quality. Run a live question with your keys to validate provider access and model behavior.

Code: `coeus/engine.py` orchestrates; `providers.py` handles hosted APIs and demo fixtures; `storage.py` persists; `models.py` defines contracts; `report.py` validates and renders; `app.py` provides Streamlit; `cli.py` runs a single job.

Provider contracts: [Tavily search](https://docs.tavily.com/documentation/api-reference/endpoint/search), [Groq chat completions](https://console.groq.com/docs/api-reference), [Streamlit uploads](https://docs.streamlit.io/develop/api-reference/widgets/st.file_uploader).
