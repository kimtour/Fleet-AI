"""Telemetry loading, validation, and tenant-scoped access.

Design rule: nothing outside this module touches the raw store. Tools get a
`TenantView`, which can only see one company's devices. A device id belonging
to another tenant is indistinguishable from a nonexistent one (no existence oracle).
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_DATA = Path(__file__).resolve().parent / "resources" / "device-telemetry-dataset.ndjson"


def parse_ts(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def form_factor(model_name: str) -> str:
    """Heuristic (NOT a telemetry field): battery presence is unreliable because
    the agent intermittently omits the battery block on laptops."""
    m = model_name.lower()
    if "mac mini" in m or "imac" in m or "mac studio" in m:
        return "desktop"
    if any(k in m for k in ("macbook", "thinkpad", "xps", "latitude", "elitebook")):
        return "laptop"
    return "unknown"


class NotFound(Exception):
    pass


@dataclass(frozen=True)
class Snapshot:
    raw: dict
    ts: datetime

    @property
    def device_id(self): return self.raw["device_id"]
    @property
    def tenant(self): return self.raw["company_id"]
    @property
    def employee_id(self): return self.raw["employee_id"]
    @property
    def ts_str(self): return self.raw["collected_at"]
    @property
    def hostname(self): return self.raw["os"]["hostname"]
    @property
    def model(self): return self.raw["device_identity"]["model_name"]
    @property
    def os_product(self): return self.raw["os"]["product_name"]
    @property
    def os_version(self): return self.raw["os"]["product_version"]
    @property
    def os_label(self): return f"{self.os_product} {self.os_version}"
    @property
    def arch(self): return self.raw["os"]["architecture"]
    @property
    def form_factor(self): return form_factor(self.model)
    @property
    def disk_total(self): return sum(v["size_bytes"] for v in self.raw["disk_volumes"])
    @property
    def disk_avail(self): return sum(v["available_bytes"] for v in self.raw["disk_volumes"])
    @property
    def disk_free_pct(self): return 100.0 * self.disk_avail / self.disk_total
    @property
    def ram_total(self): return self.raw["memory"]["total_memory_bytes"]
    @property
    def ram_used_pct(self): return 100.0 * self.raw["memory"]["used_memory_bytes"] / self.ram_total
    @property
    def battery(self): return self.raw.get("battery")  # may be missing (desktop OR data gap)
    @property
    def checks(self): return {c["check_id"]: c for c in self.raw["compliance_results"]}
    @property
    def software(self): return [s["name"] for s in self.raw["installed_software"]]


class DataStore:
    def __init__(self, path: Path | str = DEFAULT_DATA):
        self.path = Path(path)
        self.rows: list[Snapshot] = []
        self.quality: list[tuple[str | None, str]] = []   # (tenant the note may be shown to | None, text)
        self._by_device: dict[str, list[Snapshot]] = defaultdict(list)
        self._tenant_of: dict[str, str] = {}
        self._load()

    @classmethod
    def from_rows(cls, rows: list[dict]) -> "DataStore":
        self = cls.__new__(cls)
        self.path, self.rows, self.quality = Path("<memory>"), [], []
        self._by_device, self._tenant_of = defaultdict(list), {}
        self._ingest(rows)
        return self

    def _load(self):
        rows = []
        raw = self.path.read_bytes()
        self._loaded_source_hash = hashlib.sha256(raw).hexdigest()
        for i, line in enumerate(raw.decode().splitlines(), 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    self.quality.append((None, "unparseable record (skipped)"))
        self._ingest(rows)

    def _ingest(self, rows):
        seen = set()
        for r in rows:
            try:
                snap = Snapshot(r, parse_ts(r["collected_at"]))
                if not all(isinstance(r.get(k),str) and r[k].strip() for k in ('device_id','company_id','employee_id')):
                    raise ValueError('Missing identity')
                if not (snap.disk_total > 0 and 0 <= snap.disk_avail <= snap.disk_total):
                    raise ValueError('Invalid disk counters')
                if not (snap.ram_total > 0 and 0 <= r['memory']['used_memory_bytes'] <= snap.ram_total):
                    raise ValueError('Invalid memory counters')
            except Exception as e:  # noqa: BLE001 - quarantine anything malformed
                self.quality.append((r.get("company_id") if isinstance(r, dict) and isinstance(r.get("company_id"), str) else None,
                                     f"quarantined record ({type(e).__name__})"))
                continue
            did = snap.device_id
            if self._tenant_of.setdefault(did, snap.tenant) != snap.tenant:
                self.quality.append((snap.tenant, "a record was quarantined because its device id conflicts with another record"))   # never names the other tenant or its ids
                continue
            if (did, snap.ts_str) in seen:
                self.quality.append((snap.tenant, f"duplicate snapshot {did}@{snap.ts_str} (dropped)"))
                continue
            seen.add((did, snap.ts_str))
            self.rows.append(snap)
            self._by_device[did].append(snap)
        for v in self._by_device.values():
            v.sort(key=lambda s: s.ts)

    def dataset_hash(self):
        return hashlib.sha256(json.dumps([s.raw for s in self.rows],sort_keys=True).encode()).hexdigest()

    def source_hash(self):
        return getattr(self,"_loaded_source_hash",None) or self.dataset_hash()

    def current_source_hash(self):
        return hashlib.sha256(self.path.read_bytes()).hexdigest() if self.path.exists() else self.dataset_hash()

    def tenants(self) -> list[str]:
        return sorted(set(self._tenant_of.values()))

    def tenant_aliases(self) -> dict[str, set[str]]:
        """Only used by the scope guard to *refuse* cross-tenant mentions."""
        return {t: {t.lower(), t.split("-")[0].lower()} for t in self.tenants()}

    def view(self, tenant: str) -> "TenantView":
        return TenantView(self, tenant)


class TenantView:
    """The only door to telemetry for tools. Bound to a single tenant."""

    def __init__(self, store: DataStore, tenant: str):
        self._s, self.tenant = store, tenant
        self._ids, self._anchor = None, None

    def device_ids(self) -> list[str]:
        return sorted(d for d, t in self._s._tenant_of.items() if t == self.tenant and d in self._s._by_device and (self._ids is None or d in self._ids))

    def filtered(self, form_factor=None, device_id=None):
        if device_id is not None:
            self.series(device_id)  # Scope check before selecting a subset.
        result=TenantView(self._s,self.tenant)
        result._ids={d for d in self.device_ids() if (device_id is None or d==device_id)
                     and (form_factor is None or self.latest(d).form_factor==form_factor)}
        result._anchor=self.as_of
        return result

    def series(self, device_id: str) -> list[Snapshot]:
        if self._s._tenant_of.get(device_id) != self.tenant or device_id not in self._s._by_device or (self._ids is not None and device_id not in self._ids):
            raise NotFound(f"device '{device_id}' not found")
        return self._s._by_device[device_id]

    def latest(self, device_id: str) -> Snapshot:
        return self.series(device_id)[-1]

    def owns_employee(self, employee_id: str) -> bool:
        return any(self.latest(d).employee_id == employee_id for d in self.device_ids())

    def devices_of(self, employee_id: str) -> list[str]:
        return [d for d in self.device_ids() if self.latest(d).employee_id == employee_id]

    @property
    def as_of(self) -> datetime:
        """'Now' is the newest snapshot in the tenant, never the wall clock
        (the dataset ends 2026-06-12; wall-clock 'now' would make everything look stale)."""
        return self._anchor if self._anchor is not None else max(self.latest(d).ts for d in self.device_ids())

    def quality_notes(self) -> list[str]:
        notes = [t for tenant, t in self._s.quality if tenant == self.tenant]
        if (n := sum(1 for tenant, _ in self._s.quality if tenant is None)):
            notes.append(f"{n} record(s) could not be attributed to any company and were skipped")
        emp = {x.employee_id for d in self.device_ids() for x in self.series(d)}
        notes.append(f"{len(emp)} distinct employee id(s) are represented in this tenant's recorded telemetry")
        gaps = [d for d in self.device_ids() if any(s.battery is None for s in self.series(d))
                and any(s.battery is not None for s in self.series(d))]
        for d in gaps:
            n = sum(1 for s in self.series(d) if s.battery is None)
            notes.append(f"{d}: battery block missing in {n} of {len(self.series(d))} snapshots (agent data gap, not 'no battery')")
        return notes
