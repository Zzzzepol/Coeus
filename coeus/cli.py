import argparse
import asyncio
from pathlib import Path

from coeus.config import Settings
from coeus.engine import ResearchEngine
from coeus.models import Document
from coeus.providers import create_provider
from coeus.storage import Store


def main():
    parser = argparse.ArgumentParser(description="Run one inspectable Coeus research job")
    parser.add_argument("question", nargs="?")
    parser.add_argument("--mode", choices=["live", "hybrid", "offline"], default="live")
    parser.add_argument("--profile", choices=["quick", "thorough", "custom"], default="quick")
    parser.add_argument("--resume", help="Saved run ID to retry without overwriting the original")
    parser.add_argument("--preferred-domain", action="append", default=[])
    parser.add_argument(
        "--offline", action="store_true", help="Use local Ollama on uploaded documents only"
    )
    parser.add_argument("--document", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, default=Path("reports/latest.md"))
    args = parser.parse_args()
    if not args.question and not args.resume:
        parser.error("Provide a research question or --resume RUN_ID")
    if (args.offline or args.mode == "offline") and not args.document and not args.resume:
        parser.error("--offline requires at least one --document path (.txt or .md)")
    settings = Settings().with_profile(args.profile)
    settings.preferred_domains = args.preferred_domain
    store = Store(settings.db_path)
    try:
        documents = []
        for path in args.document:
            if path.suffix.lower() not in {".txt", ".md"} or path.stat().st_size > 400_000:
                raise ValueError("Documents must be .txt/.md files of at most 400 KB.")
            documents.append(Document(name=path.name, text=path.read_text(encoding="utf-8")))

        async def execute():
            mode = "offline" if args.offline else args.mode
            if args.resume:
                mode = store.get(args.resume)["mode"]
            if mode == "offline" and not documents and not args.resume:
                raise ValueError("Offline mode requires --document with a text or Markdown file.")
            provider = create_provider(settings, mode)
            try:
                return await ResearchEngine(settings, store, provider, print).run(
                    args.question or "", documents, mode, resume_from=args.resume
                )
            finally:
                await provider.close()

        run_id = asyncio.run(execute())
    except KeyboardInterrupt:
        parser.exit(130, "Research cancelled; completed artifacts remain saved.\n")
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(
            2,
            f"Cannot start research ({type(exc).__name__}). Check input files and configuration.\n",
        )
    result = store.get(run_id)
    print(f"Run: {run_id} | Status: {result['status']} | Database: {settings.db_path}")
    if result["report"]:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result["report"], encoding="utf-8")
        print(f"Report: {args.output.resolve()}")
    else:
        parser.exit(1, result["error"] + "\n")


if __name__ == "__main__":
    main()
