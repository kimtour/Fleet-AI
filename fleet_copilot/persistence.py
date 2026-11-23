"""SQLite transactions shared by proposals, simulated outbox, and audit events."""
from contextlib import contextmanager
from pathlib import Path
import sqlite3
import threading

_local = threading.local()


class Database:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.transaction() as db:
            schema = '''
              CREATE TABLE IF NOT EXISTS proposals (id TEXT PRIMARY KEY, payload TEXT NOT NULL);
              CREATE TABLE IF NOT EXISTS outbox (idempotency_key TEXT PRIMARY KEY, payload TEXT NOT NULL);
              CREATE TABLE IF NOT EXISTS audit (seq INTEGER PRIMARY KEY, record TEXT NOT NULL);
            '''
            for statement in schema.split(';'):
                if statement.strip():
                    db.execute(statement)

    @contextmanager
    def transaction(self):
        active = getattr(_local, 'active', None)
        if active is None:
            active = _local.active = {}
        key = str(self.path)
        if key in active:
            yield active[key]
            return
        db = sqlite3.connect(self.path, timeout=30)
        try:
            db.execute('PRAGMA busy_timeout=30000')
            db.execute('PRAGMA synchronous=FULL')
            db.execute('BEGIN IMMEDIATE')
            active[key] = db
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            active.pop(key, None)
            db.close()
