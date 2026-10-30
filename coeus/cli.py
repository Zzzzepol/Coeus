import argparse
import asyncio
from pathlib import Path

from coeus.config import Settings
from coeus.engine import ResearchEngine
from coeus.models import Document
from coeus.providers import DemoProvider, HostedProvider
from coeus.storage import Store


def main():
    parser = argparse.ArgumentParser(description="Run one inspectable Coeus research job")
    parser.add_argument("question")
    parser.add_argument("--demo", action="store_true", help="Use synthetic offline fixtures")
    parser.add_argument("--document", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, default=Path("reports/latest.md"))
    args = parser.parse_args()
    settings = Settings()
    store = Store(settings.db_path)
    try:
        documents = []
        for path in args.document:
            if path.suffix.lower() not in {".txt", ".md"} or path.stat().st_size > 400_000:
                raise ValueError("Documents must be .txt/.md files of at most 400 KB.")
            documents.append(Document(name=path.name, text=path.read_text(encoding="utf-8")))

        async def execute():
            provider = DemoProvider() if args.demo else HostedProvider(settings)
            try:
                return await ResearchEngine(settings, store, provider, print).run(
                    args.question, documents, "demo" if args.demo else "live"
                )
            finally:
                await provider.close()

        run_id = asyncio.run(execute())
    except (ValueError, OSError) as exc:
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
