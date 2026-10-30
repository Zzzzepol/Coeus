import asyncio

import streamlit as st
from pydantic import ValidationError

from coeus.config import Settings
from coeus.engine import ResearchEngine
from coeus.models import Document
from coeus.providers import DemoProvider, HostedProvider
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

with st.sidebar:
    st.header("Research workspace")
    mode = st.radio(
        "Run mode",
        ["demo", "live"],
        format_func=lambda m: "Offline demonstration" if m == "demo" else "Live research",
    )
    st.caption(f"Model: {settings.llm_model} · Workers: {settings.workers}")
    st.caption(
        "Documents and evidence are stored locally. Live runs send questions and uploaded "
        "document text to the configured LLM; search queries go to Tavily."
    )
    st.caption("PDF and DOCX support are deferred. Upload .txt or .md files.")

with st.form("research"):
    question = st.text_area(
        "Research question",
        placeholder="What are the tradeoffs of SQLite for a local research application?",
        max_chars=2000,
    )
    uploads = st.file_uploader(
        "Supporting documents (optional)", type=["txt", "md"], accept_multiple_files=True
    )
    submitted = st.form_submit_button("Run research", type="primary")

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
        if mode == "live":
            settings.validate_live()
        with st.status("Research in progress", expanded=True) as status:

            async def execute():
                provider = DemoProvider() if mode == "demo" else HostedProvider(settings)
                try:
                    return await ResearchEngine(settings, store, provider, st.write).run(
                        question, docs, mode
                    )
                finally:
                    await provider.close()

            run_id = asyncio.run(execute())
            st.session_state["selected_run"] = run_id
            saved = store.get(run_id)
            status.update(
                label=f"Run {saved['status'].replace('_', ' ')}",
                state="error" if saved["status"] == "failed" else "complete",
                expanded=False,
            )
    except UnicodeDecodeError:
        st.error("Documents must be UTF-8 encoded text.")
    except ValidationError:
        st.error("Documents must have a name and 1–100,000 non-empty characters.")
    except ValueError as exc:
        st.error(str(exc))

runs = store.runs()
if not runs:
    st.info("Start with the offline demonstration, or add API keys in .env for live research.")
else:
    lookup = {r["id"]: r for r in runs}
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
            "This run is running or was interrupted before completion. Saved artifacts remain "
            "available; interrupted runs are not automatically resumed."
        )
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
                st.text(source["content"])
    with plan_tab:
        st.json(artifacts.get("plan", {}))
        st.json(artifacts.get("task", {}))
        st.json(artifacts.get("config", {}))
    with critic_tab:
        st.json(artifacts.get("critique", {}))
        st.json(artifacts.get("warnings", {}))
    with events_tab:
        st.dataframe(run["events"], hide_index=True, use_container_width=True)
