"""Tool catalog. Read tools return data + evidence ids. Action tools are propose-only.

Tenant is NEVER a tool argument: it is bound in ToolContext from the authenticated
principal. Schemas reject unknown arguments, so a planner that tries to pass
`company_id`/`tenant_id` gets a hard error instead of cross-tenant access.
"""
from __future__ import annotations

import math
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable

from . import analytics as A
from .data import NotFound, TenantView, form_factor
from .evidence import AuditLog, EvidenceLedger
from .policy import PolicyEngine, Principal, ProposalService

SEV_RANK = {"critical": 0, "high": 1, "medium": 2, "watch": 3, "info": 4}


@dataclass
class ToolContext:
    view: TenantView
    ledger: EvidenceLedger
    principal: Principal
    session: str
    audit: AuditLog
    policy: PolicyEngine
    proposals: ProposalService
    th: A.Thresholds = A.Thresholds()


@dataclass
class ToolResult:
    ok: bool
    tool: str
    data: dict = field(default_factory=dict)
    evidence: list[str] = field(default_factory=list)
    error: str | None = None
    rule: str | None = None


@dataclass
class ToolSpec:
    name: str
    description: str
    kind: str                       # "read" | "action"
    params: dict[str, dict]         # name -> {type, enum?, required?, default?}
    fn: Callable[..., ToolResult]


def _validate(spec: ToolSpec, args: dict) -> tuple[dict | None, str | None]:
    unknown = set(args) - set(spec.params)
    if unknown:
        return None, f"unknown argument(s) {sorted(unknown)}; tenant scope is implicit and cannot be set"
    out = {}
    types = {"str": str, "int": int, "float": (int, float), "bool": bool}
    for name, p in spec.params.items():
        if name not in args:
            if p.get("required"):
                return None, f"missing required argument '{name}'"
            if "default" in p:
                out[name] = p["default"]
            continue
        v = args[name]
        if v is None and not p.get("required"):
            continue
        if not isinstance(v, types[p["type"]]) or (p["type"] != "bool" and isinstance(v, bool)):
            return None, f"argument '{name}' must be {p['type']}"
        if isinstance(v,(int,float)) and not isinstance(v,bool):
            if not math.isfinite(v) or (name=='threshold_pct' and not 0 <= v <= 100) or (name=='major' and v<1):
                return None, f"argument '{name}' must be finite and within its policy range"
        if "enum" in p and v not in p["enum"]:
            return None, f"argument '{name}' must be one of {p['enum']}"
        out[name] = v
    return out, None


# ------------------------------------------------------------------ read tools
def _coverage(ctx, tool, extra=None):
    v = ctx.view
    snaps = [s for d in v.device_ids() for s in v.series(d)]
    f = {"devices_examined": len(v.device_ids()), "as_of": v.as_of.strftime("%Y-%m-%dT%H:%M:%SZ"), **(extra or {})}
    return ctx.ledger.add(device_id=None, kind="coverage", ts=f["as_of"], fields=f, source=tool)


def fleet_overview(ctx: ToolContext, form_factor=None) -> ToolResult:
    v = ctx.view.filtered(form_factor=form_factor)
    latest = [v.latest(d) for d in v.device_ids()]
    snaps = [s for d in v.device_ids() for s in v.series(d)]
    f = {"devices": len(latest), "snapshots": len(snaps), "first_ts": min((s.ts_str for s in snaps),default=None),
         "last_ts": max((s.ts_str for s in snaps),default=None),
         "by_os": dict(sorted(_count(f"{s.os_product}" for s in latest).items())),
         "by_form_factor": dict(sorted(_count(s.form_factor for s in latest).items())),
         "employees": len({s.employee_id for s in latest})}
    ev = ctx.ledger.add(device_id=None, kind="coverage", ts=f"{f['first_ts']}..{f['last_ts']}", fields=f, source="fleet_overview")
    return ToolResult(True, "fleet_overview", {"overview": f, "quality_notes": v.quality_notes(), "ev": ev}, [ev])


def _count(it):
    out = {}
    for x in it:
        out[x] = out.get(x, 0) + 1
    return out


def find_low_disk(ctx: ToolContext, threshold_pct: float = 10.0, form_factor: str | None = None) -> ToolResult:
    rows, evs = [], []
    for d in ctx.view.device_ids():
        s = ctx.view.latest(d)
        if form_factor and s.form_factor != form_factor:
            continue
        if s.disk_free_pct < threshold_pct:
            ev = ctx.ledger.add(device_id=d, kind="snapshot", ts=s.ts_str, source="find_low_disk",
                                fields={"disk_free_pct": round(s.disk_free_pct, 1), "disk_free_gib": round(s.disk_avail / 2**30, 1),
                                        "disk_total_gib": round(s.disk_total / 2**30), "hostname": s.hostname})
            rows.append({"device_id": d, "hostname": s.hostname, "free_pct": round(s.disk_free_pct, 1),
                         "free_gb": round(s.disk_avail / 2**30, 1), "total_gb": round(s.disk_total / 2**30), "ev": ev})
            evs.append(ev)
    rows.sort(key=lambda r: (r["free_pct"], r["device_id"]))
    cov = _coverage(ctx, "find_low_disk", {"threshold_pct": threshold_pct, "form_factor": form_factor})
    return ToolResult(True, "find_low_disk", {"threshold_pct": threshold_pct, "devices": rows, "cov": cov}, evs + [cov])


def find_compliance_failures(ctx: ToolContext, severity: str | None = None, check_id: str | None = None,
                             form_factor: str | None = None) -> ToolResult:
    rows, evs = [], []
    for d in ctx.view.device_ids():
        series = ctx.view.series(d)
        s = series[-1]
        if form_factor and s.form_factor != form_factor:
            continue
        for cid, c in sorted(s.checks.items()):
            if c["status"] != "fail" or (severity and c["severity"] != severity) or (check_id and cid != check_id):
                continue
            h = A.check_history(series, cid)
            ev = ctx.ledger.add(device_id=d, kind="snapshot", ts=s.ts_str, source="find_compliance_failures",
                                fields={"check_id": cid, "status": "fail", "severity": c["severity"],
                                        "consecutive_failing_snapshots": h["current_fail_streak"],
                                        "first_fail_ts": h["first_fail_ts"], "hostname": s.hostname})
            rows.append({"device_id": d, "hostname": s.hostname, "check_id": cid, "severity": c["severity"],
                         "streak": h["current_fail_streak"], "first_fail_ts": h["first_fail_ts"], "ev": ev})
            evs.append(ev)
    rows.sort(key=lambda r: (r["check_id"], r["device_id"]))
    # Coverage evidence is what makes a NEGATIVE answer ("none fail") citable.
    pool = [d for d in ctx.view.device_ids() if not form_factor or ctx.view.latest(d).form_factor == form_factor]
    checks_seen = sorted({(cid, c["severity"]) for d in pool for cid, c in ctx.view.latest(d).checks.items()})
    cov = _coverage(ctx, "find_compliance_failures",
                    {"severity": severity, "check_id": check_id, "form_factor": form_factor, "devices_in_scope": len(pool),
                     "checks_present": [f"{c}:{sv}" for c, sv in checks_seen]})
    return ToolResult(True, "find_compliance_failures",
                      {"filters": {"severity": severity, "check_id": check_id, "form_factor": form_factor},
                       "failures": rows, "devices_in_scope": len(pool), "cov": cov}, evs + [cov])


def count_os_below(ctx: ToolContext, product: str, major: int) -> ToolResult:
    def major_of(s):
        try:
            return int(s.os_version.replace(".", " ").split()[0])
        except ValueError:
            return None
    older, ok, other, evs = [], [], 0, []
    for d in ctx.view.device_ids():
        s = ctx.view.latest(d)
        if s.os_product.lower() != product.lower():
            other += 1
            continue
        m = major_of(s)
        ev = ctx.ledger.add(device_id=d, kind="snapshot", ts=s.ts_str, source="count_os_below",
                            fields={"os": s.os_label, "hostname": s.hostname})
        (older if m is not None and m < major else ok).append({"device_id": d, "hostname": s.hostname, "os": s.os_label, "ev": ev})
        evs.append(ev)
    cov = _coverage(ctx, "count_os_below", {"product": product, "major": major, "other_os_excluded": other})
    return ToolResult(True, "count_os_below", {"product": product, "major": major, "older": older, "not_older": ok,
                                               "other_os_excluded": other, "cov": cov}, evs + [cov])


def get_device_history(ctx: ToolContext, device_id: str) -> ToolResult:
    try:
        series = ctx.view.series(device_id)
    except NotFound:
        return ToolResult(False, "get_device_history", error="device not found in your organisation", rule="R-TENANT")
    s, as_of = series[-1], ctx.view.as_of
    rng = f"{series[0].ts_str}..{s.ts_str}"
    disk, ram, bat = A.disk_trend(series, ctx.th), A.ram_pressure(series, ctx.th), A.battery_wear(series, as_of, ctx.th)
    checks = {cid: A.check_history(series, cid) for cid in s.checks}
    ev = {
        "disk": ctx.ledger.add(device_id=device_id, kind="series", ts=rng, fields=disk, source="get_device_history:disk"),
        "ram": ctx.ledger.add(device_id=device_id, kind="series", ts=rng, fields=ram, source="get_device_history:ram"),
        "battery": ctx.ledger.add(device_id=device_id, kind="series", ts=rng, fields=bat, source="get_device_history:battery"),
        "checks": ctx.ledger.add(device_id=device_id, kind="series", ts=rng, fields=checks, source="get_device_history:checks"),
    }
    ident = {"hostname": s.hostname, "model": s.model, "os": s.os_label, "employee_id": s.employee_id, "form_factor": s.form_factor}
    return ToolResult(True, "get_device_history", {"device_id": device_id, "identity": ident, "disk": disk, "ram": ram,
                                                  "battery": bat, "checks": checks, "ev": ev}, list(ev.values()))


def detect_insights(ctx: ToolContext, kind: str, check_id=None, form_factor=None, device_id=None, severity=None) -> ToolResult:
    v, th, as_of = ctx.view.filtered(form_factor,device_id), ctx.th, ctx.view.as_of
    findings, evs = [], []

    def add(device_id, sev, title, fields, ts):
        ev = ctx.ledger.add(device_id=device_id, kind="series", ts=ts, fields=fields, source=f"detect_insights:{kind}")
        findings.append({"device_id": device_id, "severity": sev, "title": title, "fields": fields, "ev": ev})
        evs.append(ev)

    extra = {}
    for d in v.device_ids():
        series = v.series(d)
        rng = f"{series[0].ts_str}..{series[-1].ts_str}"
        host = series[-1].hostname
        if kind == "disk":
            t = A.disk_trend(series, th)
            lvl = ("critical" if t["latest_free_pct"] <= th.disk_critical_pct else
                   "high" if t["latest_free_pct"] < th.disk_low_pct else "watch" if t["latest_free_pct"] < 20 else None)
            if lvl:
                add(d, lvl, host, t, rng)
        elif kind == "ram":
            r = A.ram_pressure(series, th)
            if r["tier"] != "none":
                add(d, "high" if r["tier"] == "consistent" else "watch", host, r, rng)
        elif kind == "battery":
            w = A.battery_wear(series, as_of, th)
            if series[-1].form_factor == "laptop" and w["n_missing"]:
                extra.setdefault("gaps", []).append({"device_id": d, "missing": w["n_missing"], "of": w["n"]})
            if w.get("has_battery_data") and (w["condition_latest_known"] == "Service Recommended" or w["declining"]):
                add(d, "high" if w["service_days"] >= th.replacement_battery_days and w["declining"] else "watch", host, w, rng)
    if kind == "compliance_drift":
        for cid in sorted({c for d in v.device_ids() for c,check in v.latest(d).checks.items() if (check_id is None or c==check_id) and (severity is None or check["severity"]==severity)}):
            wins = fleet_drift = A.fleet_check_drift(v, cid)
            cov = ctx.ledger.add(device_id=None, kind="coverage", ts=as_of.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                 fields={"check_id": cid, "weekly_failing": [w["failing"] for w in wins],
                                         "weekly_seen": [w["devices_seen"] for w in wins]}, source="detect_insights:compliance_drift")
            evs.append(cov)
            reg, chronic = [], []
            for d in v.device_ids():
                series = v.series(d)
                h = A.check_history(series, cid)
                if not h["n"]:
                    continue
                rng = f"{series[0].ts_str}..{series[-1].ts_str}"
                if h["regression_ts"] and h["latest"] == "fail":
                    ev = ctx.ledger.add(device_id=d, kind="series", ts=rng, fields={"check_id": cid, **h}, source="detect_insights:compliance_drift")
                    reg.append({"device_id": d, "hostname": series[-1].hostname, "regressed_at": h["regression_ts"], "streak": h["current_fail_streak"], "ev": ev})
                    evs.append(ev)
                elif h["always_failing"]:
                    chronic.append(d)
            findings.append({"check_id": cid, "weekly": wins, "regressions": reg, "chronic": chronic, "ev": cov})
    if kind != "compliance_drift":
        findings.sort(key=lambda f: (SEV_RANK[f["severity"]], f["device_id"]))
    cov = _coverage(ctx, f"detect_insights:{kind}", {"devices_examined":len(v.device_ids()),"check_id":check_id,"form_factor":form_factor,"device_id":device_id,"severity":severity,"kind": kind, "thresholds": {k: getattr(th, k) for k in th.__dataclass_fields__}})
    return ToolResult(True, "detect_insights", {"kind": kind, "findings": findings, "cov": cov, **extra}, evs + [cov])


# ------------------------------------------------------------------ action tools (propose-only)
def _propose(ctx: ToolContext, action: str, args: dict) -> ToolResult:
    dec = ctx.policy.check(action, args)
    ctx.audit.append("guardrail_decision", tenant=ctx.principal.tenant, actor="policy", session=ctx.session,
                     action=action, args=args, allowed=dec.allowed, rule=dec.rule, reason=dec.reason)
    if not dec.allowed:
        return ToolResult(False, action, {"alternatives": dec.alternatives, "facts": dec.facts, "args": args},
                          error=dec.reason, rule=dec.rule)
    prop, dec2 = ctx.proposals.create(ctx.principal, ctx.session, action, args, dec)
    if prop is None:
        return ToolResult(False, action, {"args": args}, error=dec2.reason, rule=dec2.rule)
    dev = args.get("device_id")
    ev = ctx.ledger.add(device_id=dev, kind="series", ts=str(dec.facts.get("latest_ts", "")), fields=dec.facts,
                        source=f"policy:{action}")
    return ToolResult(True, action, {"proposal_id": prop.id, "args": args, "facts": dec.facts,
                                    "duplicate": dec2.rule == "R-DUPLICATE", "ev": ev}, [ev])


def create_upgrade_order(ctx, device_id, component, spec): return _propose(ctx, "create_upgrade_order", dict(device_id=device_id, component=component, spec=spec))
def open_remediation_ticket(ctx, device_id, check_id, note): return _propose(ctx, "open_remediation_ticket", dict(device_id=device_id, check_id=check_id, note=note))
def flag_device_for_replacement(ctx, device_id, reason, basis=None): return _propose(ctx, "flag_device_for_replacement", dict(device_id=device_id, reason=reason, **({"basis":basis} if basis else {})))
def notify_employee(ctx, employee_id, message): return _propose(ctx, "notify_employee", dict(employee_id=employee_id, message=message))


S = lambda **k: {"type": "str", **k}  # noqa: E731
REGISTRY: dict[str, ToolSpec] = {t.name: t for t in [
    ToolSpec("fleet_overview", "Counts of devices/snapshots/OS/form factors, data-quality notes.", "read", {"form_factor":S(enum=["laptop","desktop"])}, fleet_overview),
    ToolSpec("find_low_disk", "Devices whose latest free disk % is below a threshold.", "read",
             {"threshold_pct": {"type": "float", "default": 10.0}, "form_factor": S(enum=["laptop", "desktop"])}, find_low_disk),
    ToolSpec("find_compliance_failures", "Checks failing in each device's latest snapshot, filterable by severity/check/form factor.", "read",
             {"severity": S(enum=["low", "medium", "high"]), "check_id": S(), "form_factor": S(enum=["laptop", "desktop"])}, find_compliance_failures),
    ToolSpec("count_os_below", "Devices of an OS product (e.g. macOS) whose major version is below N.", "read",
             {"product": S(required=True), "major": {"type": "int", "required": True}}, count_os_below),
    ToolSpec("get_device_history", "30-day disk/RAM/battery/compliance history for one device.", "read",
             {"device_id": S(required=True)}, get_device_history),
    ToolSpec("detect_insights", "Trend detectors: disk, ram, battery, compliance_drift.", "read",
             {"kind": S(required=True, enum=["disk", "ram", "battery", "compliance_drift"]),"check_id":S(),"form_factor":S(enum=["laptop","desktop"]),"device_id":S(),"severity":S(enum=["high","medium","low"])}, detect_insights),
    ToolSpec("create_upgrade_order", "PROPOSE a RAM/disk upgrade order (needs approval).", "action",
             {"device_id": S(required=True), "component": S(required=True, enum=["disk", "ram"]), "spec": S(required=True)}, create_upgrade_order),
    ToolSpec("open_remediation_ticket", "PROPOSE a remediation ticket for a failing check (needs approval).", "action",
             {"device_id": S(required=True), "check_id": S(required=True), "note": S(required=True)}, open_remediation_ticket),
    ToolSpec("flag_device_for_replacement", "PROPOSE flagging a device for replacement (needs approval).", "action",
             {"device_id": S(required=True), "reason": S(required=True), "basis": S(enum=["battery","disk"])}, flag_device_for_replacement),
    ToolSpec("notify_employee", "PROPOSE a message to an employee (needs approval).", "action",
             {"employee_id": S(required=True), "message": S(required=True)}, notify_employee),
]}


def call_tool(ctx: ToolContext, name: str, args: dict) -> ToolResult:
    spec = REGISTRY.get(name)
    if spec is None:
        res = ToolResult(False, name, error=f"unknown tool '{name}'", rule="R-UNKNOWN-TOOL")
    else:
        clean, err = _validate(spec, args)
        res = ToolResult(False, name, error=err, rule="R-ARGS") if err else spec.fn(ctx, **clean)
    digest = hashlib.sha256(json.dumps(res.data, sort_keys=True, default=str).encode()).hexdigest()[:16]
    ctx.audit.append("tool_call", tenant=ctx.principal.tenant, actor="agent", session=ctx.session, tool=name,
                     args=args, ok=res.ok, error=res.error, rule=res.rule, evidence=res.evidence, result_digest=digest)
    return res


def catalog() -> list[dict]:
    return [{"name": t.name, "kind": t.kind, "description": t.description, "params": t.params} for t in REGISTRY.values()]
