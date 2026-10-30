import streamlit as st
from pydantic import ValidationError

from coeus.config import Settings
from coeus.jobs import active_job, start_job
from coeus.models import Document
from coeus.storage import Store

st.set_page_config(page_title="Coeus · Research", page_icon="🔎", layout="wide")
st.title("Coeus")
st.caption("Ask a question. Follow the evidence. Inspect the report.")
try:
    settings = Settings()
    store = Store(settings.db_path)
except (ValidationError, OSError) as exc:
    st.error(f"Cannot initialize Coeus ({type(exc).__name__}). Check .env and the database path.")
    st.stop()

job = st.session_state.get("research_job") or active_job(settings.db_path)
if job and job.done.is_set():
    result = job.snapshot()
    if result["run_id"]:
        st.session_state["selected_run"] = result["run_id"]
    if result["error"]:
        st.error(result["error"])
    if result["cancelled"]:
        st.info("Research cancelled. Completed artifacts are saved.")
    st.session_state.pop("research_job", None)
    job = None

with st.sidebar:
    st.header("Research workspace")
    mode = st.radio(
        "Run mode",
        ["offline", "hybrid", "live"],
        format_func=lambda m: (
            "Offline · local Ollama + documents"
            if m == "offline"
            else "Hybrid · local Ollama + web search"
            if m == "hybrid"
            else "Live · hosted AI + web search"
        ),
    )
    st.caption(f"Model: {settings.llm_model if mode == 'live' else settings.ollama_model}")
    if mode != "live":
        settings.ollama_model = st.text_input("Ollama model", value=settings.ollama_model)
        st.caption(
            "Uses Ollama locally. Offline requires documents and uses no API keys. "
            "Hybrid sends search queries to Tavily and processes results locally."
        )
    else:
        st.caption("Questions and document text go to the hosted LLM; search queries go to Tavily.")
    profile = st.radio(
        "Research depth",
        ["quick", "thorough"],
        format_func=lambda p: (
            "Quick · 2 tasks, 2 sources/task"
            if p == "quick"
            else "Thorough · 5 tasks, 4 sources/task"
        ),
    )
    settings = settings.with_profile(profile)
    preferred = st.text_input(
        "Preferred source domains (optional)", placeholder="sqlite.org, docs.python.org"
    )
    settings.preferred_domains = [d.strip().lower() for d in preferred.split(",") if d.strip()]
    st.caption("Domain preferences are ranking hints, not proof that a source is correct.")
    st.caption("PDF and DOCX support are deferred. Upload .txt or .md files.")

with st.form("research"):
    question = st.text_area(
        "Research question",
        placeholder="What are the tradeoffs of SQLite for a local research application?",
        max_chars=2000,
    )
    uploads = st.file_uploader(
        "Research documents (required offline)", type=["txt", "md"], accept_multiple_files=True
    )
    submitted = st.form_submit_button("Run research", type="primary", disabled=job is not None)

if submitted:
    try:
        if not 5 <= len(question.strip()) <= 2000:
            raise ValueError("Enter a question between 5 and 2,000 characters.")
        if len(uploads) > 5:
            raise ValueError("Upload at most five documents.")
        docs = []
        for upload in uploads:
            if upload.size > 400_000:
                raise ValueError("Each document must fit within 400 KB and 100,000 characters.")
            docs.append(Document(name=upload.name, text=upload.getvalue().decode("utf-8")))
        if mode == "offline" and not docs:
            raise ValueError("Upload at least one text or Markdown document for offline research.")
        st.session_state["research_job"] = start_job(settings, question, docs, mode)
        st.rerun()
    except UnicodeDecodeError:
        st.error("Documents must be UTF-8 encoded text.")
    except ValidationError:
        st.error("Documents must have a name and 1–100,000 non-empty characters.")
    except ValueError as exc:
        st.error(str(exc))

if job:
    st.session_state["research_job"] = job

    @st.fragment(run_every=1)
    def show_progress():
        snapshot = job.snapshot()
        if snapshot["done"]:
            st.rerun()
        if snapshot["run_id"]:
            progress = store.progress(snapshot["run_id"])
            st.progress(
                progress["finished"] / max(1, progress["total"]),
                text=f"Research tasks: {progress['finished']}/{progress['total']}",
            )
            columns = st.columns(2)
            columns[0].metric("Saved sources", progress["sources"])
            columns[1].metric("Verified findings", progress["evidence"])
            if progress["events"]:
                st.info(progress["events"][-1]["message"])
            with st.expander("Recent activity"):
                for event in progress["events"]:
                    st.text(event["message"])
        else:
            st.info("Starting research…")
        if st.button("Cancel research", disabled=job.cancel_requested.is_set()):
            job.cancel()
            st.info("Cancellation requested; saving completed work.")

    show_progress()

runs = [run for run in store.runs() if run["mode"] != "demo"]
if not runs:
    st.info(
        "Start Ollama and upload documents for offline research, or configure API keys for live research."
    )
else:
    lookup = {r["id"]: r for r in runs}
    if st.session_state.get("selected_run") not in lookup:
        st.session_state.pop("selected_run", None)
    selected = st.selectbox(
        "Saved research runs",
        list(lookup),
        key="selected_run",
        format_func=lambda k: (
            f"{lookup[k]['question'][:80]} · {lookup[k]['mode']} · {lookup[k]['status']} · {lookup[k]['created_at'][:16]}"
        ),
    )
    run = store.get(selected)
    if run["error"]:
        st.error(run["error"])
    if run["status"] == "running":
        st.warning(
            "This run is active or was interrupted. Saved work can be resumed after the active job stops."
        )
    if st.button(
        "Retry incomplete work / regenerate report",
        disabled=job is not None,
        help="Creates a new attempt using this run's saved sources, evidence and checkpoints.",
    ):
        st.session_state["research_job"] = start_job(
            settings, run["question"], mode=run["mode"], resume_from=selected
        )
        st.rerun()
    parent = run["artifacts"].get("attempt", {}).get("parent")
    if parent:
        st.caption(f"Retry of run {parent['run_id'][:8]}; the original report is preserved.")
    coverage = run["artifacts"].get("coverage", {}).get("research")
    if coverage:
        st.caption(
            f"Research coverage: {coverage['status']} · "
            f"{coverage['completed']}/{coverage['total']} tasks completed · "
            f"{coverage['with_evidence']} with verified evidence"
        )
        if coverage["status"] == "partial":
            st.warning(
                "Research coverage is partial. A passing claim review applies only to included claims."
            )
    if run["status"] == "needs_review":
        st.warning(
            "The critic still has concerns. Read the critique before relying on this report."
        )
    report_tab, evidence_tab, sources_tab, plan_tab, critic_tab, events_tab = st.tabs(
        ["Report", "Evidence", "Sources", "Plan & tasks", "Critic", "Run log"]
    )
    artifacts = run["artifacts"]
    with report_tab:
        if run["report"]:
            st.download_button(
                "Download Markdown", run["report"], f"coeus-{selected[:8]}.md", "text/markdown"
            )
            st.markdown(run["report"])
        else:
            st.info("No completed report for this run.")
        with st.expander("Structured drafts"):
            st.json(artifacts.get("report", {}))
    with evidence_tab:
        for e in artifacts.get("evidence", {}).values():
            with st.expander(f"{e['id']} · {e['claim'][:100]}"):
                st.text(e["claim"])
                st.text(e["quote"])
                st.caption(f"Source: {e['source_id']} · Task: {e['task_id']}")
    with sources_tab:
        for source in artifacts.get("source", {}).values():
            with st.expander(f"{source['id']} · {source['title']}"):
                st.text(source["uri"])
                st.caption(
                    f"Source hint: {source.get('authority_hint', 'unverified')} · "
                    f"Published: {source.get('published_date') or 'unknown'} · "
                    f"Retrieved: {source.get('retrieved_at') or 'unknown'}"
                )
                st.text(source["content"])
    with plan_tab:
        st.json(artifacts.get("plan", {}))
        st.json(artifacts.get("task", {}))
        st.json(artifacts.get("config", {}))
    with critic_tab:
        st.json(artifacts.get("critique", {}))
        st.json(artifacts.get("warnings", {}))
        st.json(artifacts.get("excluded", {}))
    with events_tab:
        st.dataframe(run["events"], hide_index=True, use_container_width=True)
