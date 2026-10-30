import json
import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4


def now() -> str:
    return datetime.now(UTC).isoformat()


class Store:
    """Short transactions; no connection is shared across workers or Streamlit reruns."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self.connect()) as db, db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY, question TEXT NOT NULL, mode TEXT NOT NULL,
                    status TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                    report TEXT, error TEXT
                );
                CREATE TABLE IF NOT EXISTS artifacts (
                    run_id TEXT NOT NULL REFERENCES runs(id), kind TEXT NOT NULL,
                    id TEXT NOT NULL, payload TEXT NOT NULL,
                    PRIMARY KEY (run_id, kind, id)
                );
                CREATE TABLE IF NOT EXISTS events (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL REFERENCES runs(id),
                    created_at TEXT NOT NULL, message TEXT NOT NULL
                );
            """)

    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        return db

    def create_run(self, question: str, mode: str) -> str:
        run_id = uuid4().hex
        with closing(self.connect()) as db, db:
            db.execute(
                "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, NULL, NULL)",
                (run_id, question, mode, "running", now(), now()),
            )
        return run_id

    def finish(self, run_id: str, status: str, report=None, error=None):
        with closing(self.connect()) as db, db:
            db.execute(
                "UPDATE runs SET status=?, report=?, error=?, updated_at=? WHERE id=?",
                (status, report, error, now(), run_id),
            )

    def put(self, run_id: str, kind: str, key: str, value):
        payload = value.model_dump() if hasattr(value, "model_dump") else value
        with closing(self.connect()) as db, db:
            db.execute(
                "INSERT OR REPLACE INTO artifacts VALUES (?, ?, ?, ?)",
                (run_id, kind, key, json.dumps(payload, ensure_ascii=False)),
            )

    def event(self, run_id: str, message: str):
        with closing(self.connect()) as db, db:
            db.execute(
                "INSERT INTO events(run_id, created_at, message) VALUES (?, ?, ?)",
                (run_id, now(), message),
            )

    def runs(self):
        with closing(self.connect()) as db:
            return [
                dict(r) for r in db.execute("SELECT * FROM runs ORDER BY created_at DESC LIMIT 100")
            ]

    def get(self, run_id: str):
        with closing(self.connect()) as db:
            row = db.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
            if row is None:
                raise KeyError(run_id)
            result = dict(row)
            result["artifacts"] = {}
            for r in db.execute(
                "SELECT kind, id, payload FROM artifacts WHERE run_id=? ORDER BY kind,id", (run_id,)
            ):
                result["artifacts"].setdefault(r["kind"], {})[r["id"]] = json.loads(r["payload"])
            result["events"] = [
                dict(r)
                for r in db.execute(
                    "SELECT created_at, message FROM events WHERE run_id=? ORDER BY seq", (run_id,)
                )
            ]
            return result
