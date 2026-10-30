# Coeus

A personal research app: question → bounded research tasks → SQLite evidence → cited Markdown → critic and optional revision. Inspect sources, exact quotes, task coverage, and claim review in Streamlit.

## Run locally

Requires Python 3.11+.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
# First setup only; do not overwrite an existing .env containing your keys:
cp -n .env.example .env
streamlit run app.py --server.address 127.0.0.1
```

## Offline research with Ollama

Select **Offline · local Ollama + documents**. Upload at least one UTF-8 `.txt` or `.md` document and enter a question about it. The local model extracts evidence, writes the report, and critiques it. No Tavily or hosted LLM calls occur in this mode, and no API keys are required.

Start the Ollama app or run `ollama serve` in another terminal. Run `ollama list` to find the exact installed model tag, then set `OLLAMA_MODEL` in `.env` or enter it in the sidebar. The default is `qwen2.5-coder:3b`. To use Qwen 3.5 4B, install that model separately and use its exact tag from `ollama list`. Coeus never downloads models automatically.

```sh
coeus 'How does Coeus store and verify research evidence?' --offline --document README.md --output reports/local-research.md
```

Offline mode uses a loopback Ollama address, rejects advertised cloud models, disables HTTP environment proxies, and runs one worker to limit local memory pressure. Uploaded documents are the only research source; model memory is not accepted as evidence. Smaller models can produce weaker analysis or fail structured-output validation. Review the quotes and critic feedback. Local generation may take longer; `OLLAMA_TIMEOUT` controls each request and `COEUS_RUN_TIMEOUT` bounds the whole run.

## Live web research

Set `TAVILY_API_KEY` and `LLM_API_KEY` in `.env`, then select **Live · hosted AI + web search**. The default hosted model is `openai/gpt-oss-120b` on Groq. Other OpenAI-compatible chat-completions endpoints supporting JSON output can be configured with `LLM_BASE_URL` and `LLM_MODEL`.

```sh
coeus 'What are the tradeoffs of SQLite for a personal research app?' --output reports/research.md
```

Live mode sends questions, uploaded document content, and evidence to the configured hosted LLM; Tavily receives search queries. Provider quotas and model availability can affect runs.

## Evidence and report behavior

- Pydantic validates outputs. Extracted quotes must match saved source content exactly. Every report citation must reference known evidence. These checks establish provenance, not factual truth.
- Research coverage is separate from claim review. Failed or partial tasks, missing evidence, rejected quotes, or omitted evidence produce partial coverage even when the critic passes. Coverage describes planned tasks, not exhaustive knowledge of a topic.
- Reports use numbered citations, a direct answer when provided by the writer, findings, limitations, and an optional diagnostics appendix. Stable evidence IDs remain in SQLite.
- The critic can trigger one revision and a second review. Remaining concerns produce `needs_review`. Failed runs retain completed artifacts for inspection.
- Default live limits: 3 workers, 4 search tasks, 3 results per task. Up to 5 documents, each at most 100,000 characters / 400 KB. Only text and Markdown are supported.
- Extraction sends one source excerpt at a time, at most 4,000 characters by default. Web snapshots are capped at 12,000 characters. HTTP 413 permits two levels of lossless excerpt splitting (at most 7 attempts per initial excerpt); successful excerpts survive failed siblings.
- Report evidence has an 8,000-byte default budget. A rejected draft request may halve that evidence up to twice; omissions are disclosed. Review calls include cited evidence only. All evidence remains saved locally.
- The default output allowance is 2,048 tokens. Tune `COEUS_EXTRACT_CHUNK_CHARS`, `COEUS_EVIDENCE_BUDGET_BYTES`, and `COEUS_LLM_MAX_OUTPUT_TOKENS` for the model. The run deadline defaults to 600 seconds.
- One foreground run executes per Streamlit session. Interrupted runs are not automatically resumed. PDF/DOCX, deployment, and distributed infrastructure are outside this MVP.

## Local data and public GitHub sharing

Runs are stored in `data/coeus.sqlite3`. Database contents and downloaded reports are plaintext. Keep this personal, unauthenticated app bound to localhost.

`.env.example` contains empty key placeholders and can be committed. `.env`, local environment variants, Streamlit secrets, SQLite files, `data/`, `reports/`, and `uploads/` are ignored by Git. Keep private documents outside the repository or inside `uploads/`. Review `git diff --cached` before publishing.

The synthetic demo feature has been removed. Existing demo records remain in the database for history but are hidden in the UI. Your real research records and reports are preserved.

## Development

```sh
python -m pytest
ruff check .
```

Deterministic providers live only under `tests/`; tests do not spend API credits or require Ollama. Production providers are hosted AI and local Ollama. See `coeus/engine.py` for orchestration, `extraction.py` for bounded extraction, `providers.py` for APIs, and `report.py` for citation validation and rendering.

API references: [Ollama](https://github.com/ollama/ollama/blob/main/docs/api.md), [Tavily](https://docs.tavily.com/documentation/api-reference/endpoint/search), [Groq](https://console.groq.com/docs/api-reference).
