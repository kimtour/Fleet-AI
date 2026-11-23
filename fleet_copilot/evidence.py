"""Evidence ledger (grounding) and tamper-evident audit log (accountability)."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .persistence import Database


@dataclass(frozen=True)
class Evidence:
    id: str
    tenant: str
    device_id: str | None
    kind: str                 # snapshot | series | coverage
    ts: str                   # snapshot timestamp, or "first..last" for a series
    fields: dict
    source: str               # tool + derivation that produced it

    def cite(self) -> str:
        return f"[{self.id}]"


class EvidenceLedger:
    """Every claim in an answer must point at an entry here. Entries are created only
    by tools, from tenant-scoped data. Deduplicated so the same fact keeps one id."""

    def __init__(self, tenant: str):
        self.tenant, self._by_key, self._by_id = tenant, {}, {}

    def add(self, *, device_id, kind, ts, fields, source) -> str:
        key = (device_id, kind, ts, json.dumps(fields, sort_keys=True, default=str))
        if key not in self._by_key:
            ev = Evidence(f"E{len(self._by_id) + 1}", self.tenant, device_id, kind, ts, fields, source)
            self._by_key[key], self._by_id[ev.id] = ev.id, ev
        return self._by_key[key]

    def get(self, eid: str) -> Evidence | None:
        return self._by_id.get(eid)

    def all(self) -> list[Evidence]:
        return list(self._by_id.values())


def _h(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


class AuditLog:
    """Hash-chained events in the same SQLite transaction as the approved action.

    Detects modification, reordering and interior deletion. Detecting tail truncation
    or a complete rewrite requires a trusted external anchor, outside this local demo.
    """
    GENESIS = '0' * 64

    def __init__(self, path, clock=None):
        self.path = Path(path)
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.db = Database(self.path.parent / 'fleet.db')

    def _tail(self):
        with self.db.transaction() as db:
            row = db.execute('SELECT record FROM audit ORDER BY seq DESC LIMIT 1').fetchone()
            if row:
                last = json.loads(row[0])
                return last['seq'], last['hash']
            return 0, self.GENESIS

    def append(self, event, *, tenant, actor, session, **payload):
        with self.db.transaction() as db:
            seq, prev = self._tail()
            rec = {'seq':seq+1, 'ts':self.clock().strftime('%Y-%m-%dT%H:%M:%SZ'),
                   'event':event, 'tenant':tenant, 'actor':actor, 'session':session,
                   'payload':payload, 'prev':prev}
            rec['hash'] = _h(rec)
            db.execute('INSERT INTO audit VALUES (?,?)', (seq+1,json.dumps(rec,sort_keys=True,default=str)))
            return rec

    def records(self):
        with self.db.transaction() as db:
            return [json.loads(r[0]) for r in db.execute('SELECT record FROM audit ORDER BY seq')]

    def verify(self):
        prev = self.GENESIS
        for i, rec in enumerate(self.records(), 1):
            body = {k:v for k,v in rec.items() if k != 'hash'}
            if rec['seq'] != i or rec['prev'] != prev or _h(body) != rec['hash']:
                return False, f'chain broken at record {i}'
            prev = rec['hash']
        return True, 'ok'
