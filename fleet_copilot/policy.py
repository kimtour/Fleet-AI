"""Guardrails for state-changing actions.

Key idea: the agent *proposes*; this module independently *re-derives the evidence from
source telemetry* and decides. Evidence supplied (or hallucinated) by a planner is never
trusted. The agent has no code path to approve or execute: only `ProposalService.approve`,
called by an authenticated human principal, can do that.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path

from . import analytics as A
from .data import DataStore, NotFound, TenantView
from .evidence import AuditLog
from .persistence import Database
from contextlib import contextmanager

URL_RE = re.compile(r"(https?://|www\.|\b[\w-]+\.(com|net|org|io|ru|xyz)\b)", re.I)
CTRL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
SIZE_RE = re.compile(r"^\s*(\d+)\s*(GB|TB)\s*$", re.I)
MAX_PROPOSALS_PER_SESSION = 25


@dataclass(frozen=True)
class Principal:
    admin_id: str
    tenant: str
    role: str = "viewer"   # viewer: can ask + propose; approver: can also approve/reject


@dataclass
class Decision:
    allowed: bool
    rule: str
    reason: str
    facts: dict = field(default_factory=dict)
    alternatives: list[str] = field(default_factory=list)


def deny(rule, reason, alternatives=None, facts=None) -> Decision:
    return Decision(False, rule, reason, facts or {}, alternatives or [])


# External knowledge (NOT telemetry): which components can physically be upgraded.
# Kept explicit so reviewers can challenge or replace it.
def upgradable(snap, component: str) -> bool:
    if snap.os_product == "macOS" and snap.arch == "arm64":
        return False                      # Apple silicon: soldered RAM and storage
    if component == "disk":
        return True
    return "xps" in snap.model.lower()    # ThinkPad X1 Carbon RAM is soldered; XPS 15 is socketed


def _clean_text(text: str, limit: int) -> Decision | None:
    if not isinstance(text, str) or not text.strip():
        return deny("R-CONTENT", "text must be a non-empty string")
    if len(text) > limit:
        return deny("R-CONTENT", f"text exceeds {limit} characters")
    if CTRL_RE.search(text) or URL_RE.search(text):
        return deny("R-CONTENT", "text may not contain links or control characters")
    return None


class PolicyEngine:
    def __init__(self, view: TenantView, th: A.Thresholds = A.Thresholds()):
        self.view, self.th = view, th

    def check(self, action: str, args: dict) -> Decision:
        try:
            return getattr(self, f"_{action}")(**args)
        except NotFound:
            # identical text for "other tenant" and "does not exist"
            return deny("R-TENANT", "target not found in your organisation")
        except TypeError as e:
            return deny("R-ARGS", f"invalid arguments: {e}")
        except AttributeError:
            return deny("R-UNKNOWN-ACTION", f"unknown action '{action}'")

    # -- open_remediation_ticket -------------------------------------------------
    def _open_remediation_ticket(self, device_id, check_id, note):
        s = self.view.series(device_id)
        latest = s[-1]
        if (bad := _clean_text(note, 500)):
            return bad
        if check_id not in latest.checks:
            return deny("R-CHECK-UNKNOWN", f"device does not report check '{check_id}'")
        h = A.check_history(s, check_id)
        facts = {"check_id": check_id, "severity": latest.checks[check_id]["severity"],
                 "latest_status": h["latest"], "consecutive_failing_snapshots": h["current_fail_streak"],
                 "first_fail_ts": h["first_fail_ts"], "latest_ts": latest.ts_str}
        if h["latest"] != "fail":
            return deny("R-NO-FAILURE", f"'{check_id}' is passing in the latest snapshot ({latest.ts_str})", facts=facts)
        if h["current_fail_streak"] < 2:
            return deny("R-TRANSIENT", "failure seen in only one snapshot; wait for it to persist", facts=facts)
        return Decision(True, "R-OK", "check failing persistently", facts)

    # -- create_upgrade_order -----------------------------------------------------
    def _create_upgrade_order(self, device_id, component, spec):
        s = self.view.series(device_id)
        latest = s[-1]
        if component not in ("disk", "ram"):
            return deny("R-COMPONENT", "component must be 'disk' or 'ram'")
        if not upgradable(latest, component):
            return deny("R-NOT-UPGRADABLE", f"{latest.model} ({latest.arch}) cannot have {component} upgraded in place",
                        alternatives=["flag_device_for_replacement"])
        if component == "disk":
            t = A.disk_trend(s, self.th)
            facts = {"latest_free_pct": t["latest_free_pct"], "slope_pp_per_day": t["slope_pp_per_day"],
                     "days_to_full": t["days_to_full"], "current_gb": t["total_gb"], "latest_ts": latest.ts_str}
            ok = t["latest_free_pct"] < self.th.disk_low_pct and (t["declining"] or t["plateau_snapshots"] >= 3)
            cur_gb = latest.disk_total / 2**30
        else:
            t = A.ram_pressure(s, self.th)
            facts = {"share_snapshots_above_threshold": t["share_above"], "mean_used_pct": t["mean_pct"],
                     "current_gb": t["ram_gb"], "latest_ts": latest.ts_str}
            ok = t["tier"] == "consistent"
            cur_gb = latest.ram_total / 2**30
        if not ok:
            return deny("R-EVIDENCE-UPGRADE", f"telemetry does not show sustained {component} constraint", facts=facts)
        m = SIZE_RE.match(str(spec))
        if not m:
            return deny("R-SPEC", "spec must look like '1TB' or '32GB'", facts=facts)
        want = int(m.group(1)) * (1024 if m.group(2).upper() == "TB" else 1)   # binary, matching telemetry sizes
        if not (cur_gb * 1.2 < want <= cur_gb * 8):
            return deny("R-SPEC", f"spec {spec} must exceed current {cur_gb:.0f}GB (and be <= 8x)", facts=facts)
        facts = {**facts, "spec_basis": "next standard capacity tier computed from telemetry; a sizing proposal, not a vendor SKU",
                 "upgradability_basis": "external hardware knowledge in policy.upgradable(), not telemetry; confirm before ordering"}
        return Decision(True, "R-OK", f"sustained {component} constraint on an upgradable device", facts)

    # -- flag_device_for_replacement ------------------------------------------------
    def _flag_device_for_replacement(self, device_id, reason, basis=None):
        if basis not in (None,'battery','disk'):
            return deny('R-ARGS','replacement basis must be battery or disk')
        s = self.view.series(device_id)
        latest = s[-1]
        if (bad := _clean_text(reason, 500)):
            return bad
        w = A.battery_wear(s, self.view.as_of, self.th)
        d = A.disk_trend(s, self.th)
        if basis != "disk" and w.get("has_battery_data") and w["service_days"] >= self.th.replacement_battery_days and w["declining"]:
            return Decision(True, "R-OK", "battery service-recommended and declining",
                            {"basis": "battery", "service_since": w["service_since"], "service_days": w["service_days"],
                             "capacity_change_pct": w["capacity_change_pct"], "latest_ts": latest.ts_str})
        if basis == 'battery':
            return deny('R-EVIDENCE-REPLACE','requested battery replacement criterion is not met')
        disk_exhausted = d["latest_free_pct"] <= self.th.disk_critical_pct and (d["declining"] or d["plateau_snapshots"] >= 3)
        if disk_exhausted and not upgradable(latest, "disk"):
            return Decision(True, "R-OK", "storage exhausted on a device whose storage cannot be upgraded",
                            {"basis": "disk_non_upgradable", "latest_free_pct": d["latest_free_pct"],
                             "plateau_snapshots": d["plateau_snapshots"], "latest_ts": latest.ts_str})
        if disk_exhausted:
            return deny("R-PROPORTIONALITY", "storage can be upgraded; propose create_upgrade_order instead",
                        alternatives=["create_upgrade_order"], facts={"latest_free_pct": d["latest_free_pct"]})
        return deny("R-EVIDENCE-REPLACE", "no replacement criterion met (battery service >= "
                    f"{self.th.replacement_battery_days}d with declining capacity, or exhausted non-upgradable storage)",
                    facts={"battery_service_days": w.get("service_days"), "latest_free_pct": d["latest_free_pct"]})

    # -- notify_employee ------------------------------------------------------------
    def _notify_employee(self, employee_id, message):
        if not self.view.owns_employee(employee_id):
            raise NotFound(employee_id)
        if (bad := _clean_text(message, 600)):
            return bad
        findings = []
        for d in self.view.devices_of(employee_id):
            s = self.view.series(d)
            for cid, c in s[-1].checks.items():
                if c["status"] == "fail":
                    findings.append(f"{d}:{cid}")
            if A.disk_trend(s, self.th)["latest_free_pct"] < self.th.disk_low_pct:
                findings.append(f"{d}:disk_low")
        if not findings:
            return deny("R-EVIDENCE-NOTIFY", "employee's devices have no open finding to notify about")
        return Decision(True, "R-OK", "employee owns devices with open findings",
                        {"open_findings": findings, "devices": self.view.devices_of(employee_id)})


# ---------------------------------------------------------------------------------
@dataclass
class Proposal:
    id: str
    tenant: str
    action: str
    args: dict
    payload_hash: str
    facts: dict
    rule: str
    proposed_for: str          # admin who asked
    session: str
    status: str = "pending"    # pending | executed | rejected | invalidated | expired
    resolution: dict = field(default_factory=dict)
    dataset_hash: str = ""
    source_hash: str = ""
    created_at: str = ""       # ISO-8601 UTC; pending proposals expire after PROPOSAL_TTL


PROPOSAL_TTL = timedelta(hours=24)


def payload_hash(tenant: str, action: str, args: dict) -> str:
    return hashlib.sha256(json.dumps({"t": tenant, "a": action, "args": args}, sort_keys=True).encode()).hexdigest()


class ProposalService:
    def __init__(self, store: "DataStore", audit: AuditLog, state_dir: Path | str, th=A.Thresholds()):
        self.store, self.audit, self.th = store, audit, th
        self.dir = Path(state_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.db = Database(self.dir / 'fleet.db')
        if any((self.dir / n).exists() for n in ('proposals.json','outbox.jsonl','audit.jsonl')):
            raise ValueError('Legacy JSON state detected; choose a new --state directory. Original files are preserved.')
        self._p = {}

    def _save(self):
        with self.db.transaction() as db:
            for key, proposal in self._p.items():
                db.execute('INSERT INTO proposals VALUES (?,?) ON CONFLICT(id) DO UPDATE SET payload=excluded.payload',
                           (key,json.dumps(asdict(proposal),sort_keys=True)))

    def _reload(self):
        with self.db.transaction() as db:
            self._p = {key:Proposal(**json.loads(raw)) for key,raw in db.execute('SELECT id,payload FROM proposals')}

    @contextmanager
    def _txn(self):
        with self.db.transaction():
            self._reload()
            yield

    # called by the agent (propose-only)
    def create(self, principal: Principal, session: str, action: str, args: dict, decision: Decision) -> tuple[Proposal | None, Decision]:
        with self._txn():
            return self._create(principal, session, action, args, decision)

    def _create(self, principal, session, action, args, decision):
        assert decision.allowed
        n = sum(1 for p in self._p.values() if p.session == session)
        if n >= MAX_PROPOSALS_PER_SESSION:
            return None, deny("R-BLAST-RADIUS", f"max {MAX_PROPOSALS_PER_SESSION} proposals per session")
        # Re-derive eligibility here as well: public callers cannot forge an allowed Decision.
        checked = PolicyEngine(self.store.view(principal.tenant),self.th).check(action,args)
        if not checked.allowed:
            return None, checked
        if self.store.source_hash() != self.store.current_source_hash():
            return None, deny('R-EVIDENCE','dataset changed; reload telemetry before proposing')
        h = payload_hash(principal.tenant, action, args)
        created_at = self.audit.clock().isoformat()
        for old in self._p.values():
            if old.payload_hash != h or old.tenant != principal.tenant:
                continue
            if old.status == 'executed':
                return None, deny('R-STATE','identical action already executed; duplicate execution is refused')
            if old.status == 'pending':
                expired = not old.created_at or self.audit.clock()-datetime.fromisoformat(old.created_at) >= PROPOSAL_TTL
                changed = old.dataset_hash != self.store.dataset_hash() or old.source_hash != self.store.source_hash()
                if not expired and not changed:
                    return old, Decision(True,'R-DUPLICATE','identical proposal already pending',checked.facts)
                old.status = 'expired' if expired else 'invalidated'
                self.audit.append('proposal_superseded',tenant=principal.tenant,actor='policy',session=session,
                                  proposal_id=old.id,reason=old.status)
        generation = hashlib.sha256((h+created_at+self.store.dataset_hash()+str(sum(p.payload_hash==h for p in self._p.values()))).encode()).hexdigest()
        pid = f'P-{generation[:16]}'
        prop = Proposal(pid,principal.tenant,action,args,h,checked.facts,checked.rule,principal.admin_id,session,
                        created_at=created_at,dataset_hash=self.store.dataset_hash(),source_hash=self.store.source_hash())
        self._p[pid] = prop
        self._save()
        self.audit.append("proposal_created", tenant=principal.tenant, actor="agent", session=session,
                          proposal_id=pid, action=action, args=args, payload_hash=h, facts=decision.facts,
                          created_at=prop.created_at,dataset_hash=prop.dataset_hash,source_hash=prop.source_hash)
        return prop, decision

    def pending(self, tenant: str) -> list[Proposal]:
        self._reload()
        return [p for p in self._p.values() if p.tenant == tenant and p.status == "pending"]

    def get(self, pid: str, principal: Principal) -> Proposal:
        self._reload()
        p = self._p.get(pid)
        if not p or p.tenant != principal.tenant:
            raise NotFound(pid)         # cross-tenant ids look nonexistent
        return p

    # called only by humans
    def approve(self, pid: str, approver: Principal) -> dict:
        with self._txn():
            return self._approve(pid, approver)

    def _approve(self, pid: str, approver: Principal) -> dict:
        def refuse(rule, why, p=None):
            self.audit.append("approval_refused", tenant=approver.tenant, actor=approver.admin_id, session="approval",
                              proposal_id=pid, rule=rule, reason=why)
            return {"ok": False, "rule": rule, "reason": why}
        try:
            p = self.get(pid, approver)
        except NotFound:
            return refuse("R-TENANT", "proposal not found")
        if approver.role != "approver" or approver.admin_id.strip().casefold().startswith("agent"):
            return refuse("R-ROLE", "principal is not an authorised human approver")
        if p.status != "pending":
            return refuse("R-STATE", f"proposal is {p.status}")
        recorded = [r for r in self.audit.records() if r["event"] == "proposal_created" and r["payload"]["proposal_id"] == pid]
        if (not recorded or recorded[-1]["payload"]["payload_hash"] != payload_hash(p.tenant, p.action, p.args)
                or p.payload_hash != recorded[-1]["payload"]["payload_hash"]):
            p.status, p.resolution = "invalidated", {"why": "payload differs from what was audited"}
            self._save()
            return refuse("R-TAMPER", "proposal payload does not match audit record")
        audited = recorded[-1]['payload']
        if any(audited.get(key) != getattr(p,key) for key in ('created_at','dataset_hash','source_hash')):
            return refuse('R-TAMPER','proposal evidence or expiry differs from the audit record')
        if not p.created_at:
            return refuse('R-EXPIRED','proposal has no verifiable creation time; propose again')
        if p.created_at and self.audit.clock() - datetime.fromisoformat(p.created_at) >= PROPOSAL_TTL:
            p.status, p.resolution = "expired", {"why": f"older than {PROPOSAL_TTL}"}
            self._save()
            return refuse("R-EXPIRED", "proposal expired (24h); ask the copilot to propose it again against current data")
        if not self.audit.verify()[0]:
            return refuse('R-TAMPER','audit chain verification failed')
        if p.dataset_hash != self.store.dataset_hash() or p.source_hash != self.store.current_source_hash():
            return refuse('R-EVIDENCE','evidence no longer sufficient: dataset changed; propose again')
        fresh = PolicyEngine(self.store.view(p.tenant), self.th).check(p.action, p.args)   # re-validate at approval time
        if not fresh.allowed:
            p.status, p.resolution = "invalidated", {"why": fresh.reason}
            self._save()
            return refuse(fresh.rule, f"evidence no longer sufficient: {fresh.reason}")
        self.audit.append("proposal_approved", tenant=p.tenant, actor=approver.admin_id, session="approval", proposal_id=pid)
        done = self._execute(p, approver)
        p.status, p.resolution = "executed", {"approved_by": approver.admin_id, **done}
        self._save()
        return {"ok": True, "proposal": pid, **done}

    def reject(self, pid: str, approver: Principal, reason: str = "") -> dict:
        with self._txn():
            return self._reject(pid, approver, reason)

    def _reject(self, pid: str, approver: Principal, reason: str = "") -> dict:
        try:
            p = self.get(pid, approver)
        except NotFound:
            return {"ok": False, "rule": "R-TENANT", "reason": "proposal not found"}
        if approver.role != "approver" or p.status != "pending":
            return {"ok": False, "rule": "R-ROLE/R-STATE", "reason": "not permitted or not pending"}
        p.status, p.resolution = "rejected", {"by": approver.admin_id, "reason": reason}
        self._save()
        self.audit.append("proposal_rejected", tenant=p.tenant, actor=approver.admin_id, session="approval",
                          proposal_id=pid, reason=reason)
        return {"ok": True, "proposal": pid, "status": "rejected"}

    def executions(self, tenant):
        with self.db.transaction() as db:
            return [r for raw, in db.execute('SELECT payload FROM outbox')
                    if (r := json.loads(raw))['tenant'] == tenant]

    def _execute(self, p, approver):
        """Local side effect and audit record commit with the approval, or all roll back."""
        with self.db.transaction() as db:
            payload = {'idempotency_key':p.payload_hash,'tenant':p.tenant,'action':p.action,
                       'args':p.args,'approved_by':approver.admin_id}
            db.execute('INSERT OR IGNORE INTO outbox VALUES (?,?)',
                       (p.payload_hash,json.dumps(payload,sort_keys=True)))
            self.audit.append('action_executed',tenant=p.tenant,actor='executor',session='approval',
                              proposal_id=p.id,action=p.action,idempotency_key=p.payload_hash)
            return {'executed':p.action,'idempotency_key':p.payload_hash[:12]}
