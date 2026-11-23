"""Planning. Two interchangeable planners behind one interface.

`RulePlanner`  - deterministic intent/slot extraction (default; makes evals reproducible).
`LLMPlanner`   - asks an LLM for a JSON plan over the tool catalog. Its output is validated
                 against the same tool schemas and falls back to RulePlanner on any problem.
In BOTH cases the LLM never sees telemetry, never writes the final answer, and cannot
bypass the policy engine: safety does not depend on planner quality.
"""
from __future__ import annotations

import json
import os
import re
import urllib.request
from dataclasses import dataclass, field

from .data import NotFound
from .tools import REGISTRY, ToolContext, _validate, catalog

DEVICE_RE = re.compile(r"\b(?=[A-Z0-9]*\d)(?=[A-Z0-9]*[A-Z])[A-Z0-9]{12}\b", re.I)
HOST_RE = re.compile(r"\b[a-z]+-(?:macbook|thinkpad|mac|dell)-\d+\b", re.I)
EMP_RE = re.compile(r"\bemp-[a-z]+-\d+\b", re.I)
CHECK_WORDS = {"screen_lock": ("screen lock", "screen-lock", "screen_lock"),
               "disk_encryption": ("encrypt",), "os_up_to_date": ("os up to date", "os_up_to_date", "up-to-date", "outdated os", "os update")}


@dataclass
class Step:
    tool: str
    args: dict
    purpose: str
    hop: int = 0                   # fallback depth (bounded)


@dataclass
class Plan:
    intent: str                    # qa_* | insights | act_* | unsupported
    steps: list[Step] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    detail: dict = field(default_factory=dict)


UNSUPPORTED_MUTATION_RE = re.compile(
    r"\b(delete|destroy|purge|wipe|erase|decommission|restart|reboot|reset|power[- ]?off|shut ?down|uninstall|format|factory[- ]?reset|re-?image|quarantine|isolate|disable|"
    r"remove|revoke|terminate|kill|lock out|lock (?:the )?(?:device|laptop|employee))\b", re.I)
NEGATED_RE = re.compile(r"\b(?:do not|don't|dont|never|without|no need to|not to)\s+(?:(?:ever|actually|please)\s+)?$",re.I)
def unsupported_mutation(question: str):
    """Honor local negation and coordinated negative verbs, but reset at a new clause."""
    previous_end, previous_negated = None, False
    for match in UNSUPPORTED_MUTATION_RE.finditer(question):
        prefix=question[:match.start()]
        negated=bool(NEGATED_RE.search(prefix))
        if previous_end is not None and previous_negated:
            bridge=question[previous_end:match.start()]
            if re.fullmatch(r"\s*(?:(?:anything|devices?|them)\s*)?(?:,?\s*(?:or|and)\s+)",bridge,re.I):
                negated=True
        if not negated:
            return match
        previous_end, previous_negated = match.end(), negated
    return None


GENERIC_PROPOSE_RE = re.compile(r"\b(propose|recommend|suggest)\w*\b.*\b(actions?|remediations?|fix(es)?|next steps?|follow[- ]ups?)\b|"
                                r"\bwhat should (we|i) do\b", re.I)


def _check_in(q):
    for cid, words in CHECK_WORDS.items():
        if any(w in q for w in words):
            return cid
    return None


def _ff(q):
    return "laptop" if re.search(r"\blaptops?\b", q) else "desktop" if re.search(r"\bdesktops?\b", q) else None


def next_tier_spec(current_gib: float) -> str:
    want = int(round(current_gib * 2))
    return f"{want // 1024}TB" if want >= 1024 and want % 1024 == 0 else f"{want}GB"


class RulePlanner:
    name = "rules"

    # ---- first-phase plan -------------------------------------------------------
    def plan(self, question: str, ctx: ToolContext) -> Plan:
        q = question.lower()
        devs = [d.upper() for d in DEVICE_RE.findall(question)]
        for h in HOST_RE.findall(question):
            for d in ctx.view.device_ids():
                if ctx.view.latest(d).hostname.lower() == h.lower():
                    devs.append(d)
        emps = EMP_RE.findall(q)
        ff = _ff(q)
        if (m := unsupported_mutation(question)):
            return Plan("unsupported", [], detail={"why": f"I can't {m.group(1).lower()} devices or settings. This copilot only answers from telemetry and "
                                                         "proposes tickets, upgrade orders, replacement reviews and employee notifications "
                                                         "for human approval; nothing was changed."})
        if re.search(r'\b(?:last|past)\s+(?:\d+\s+)?(?:days?|weeks?|months?)\b|\bbetween\s+\d{4}-\d{2}-\d{2}',q):
            return Plan('unsupported',[],detail={'why':'Custom date windows are not supported. Ask for the full recorded trend or the latest recorded snapshot.'})
        action_text=re.sub(r"\b(?:do not|don't|dont|never|without|no need to|not to)\b.*?(?=[,;.!?]|\bbut\b|\bthen\b|$)",'',q)
        act = None
        if re.search(r"\b(replac)", action_text): act = "replace"
        if re.search(r"\bupgrad|order .*(ram|disk|ssd|memory|storage)", action_text): act = "upgrade"
        if re.search(r"\b(ticket|remediat)", action_text): act = "ticket"
        if re.search(r"\b(notify|message|email|tell)\b", action_text) and (emps or "employee" in q or "owner" in q or "user" in q): act = "notify"

        if GENERIC_PROPOSE_RE.search(action_text):
            compliance = bool(_check_in(q) or re.search(r'compliance|checks?|fail|policy',q))
            battery = bool(re.search(r'batter|end[- ]of[- ]life|\beol\b',q))
            ram = bool(re.search(r'\bram\b|memory',q))
            disk = bool(re.search(r'disk|storage|space',q)) and not compliance
            if sum((compliance,battery,ram,disk)) > 1:
                return Plan('unsupported',[],detail={'why':'Please request one action area at a time: disk, RAM, battery or compliance.'})
            if battery: act = 'replace'
            elif compliance: act = 'ticket'
            elif ram or disk: act = 'upgrade'
            elif act is None:
                return Plan('unsupported',[],detail={'why':'Which area should I propose actions for: disk, RAM, battery or compliance?'})
        if act:
            plan = self._plan_action(act, q, devs, emps, ff)
            if act == 'upgrade':
                capacities=list(re.finditer(r'(?<![\w.])(-?(?:\d+(?:\.\d+)?|\.\d+))\s*(tb|gb)\b',q))
                if len(capacities)>1 and not re.search(r'\bfrom\b.*\bto\s+[-.\d]',q):
                    return Plan('unsupported',[],detail={'why':'Specify one component and one target capacity per order.'})
                if capacities:
                    sm=capacities[-1] if len(capacities)>1 else capacities[0]
                    if float(sm.group(1))<=0:
                        return Plan('unsupported',[],detail={'why':'Requested capacity must be positive; nothing was proposed.'})
                    num=sm.group(1).rstrip('0').rstrip('.') if '.' in sm.group(1) else sm.group(1)
                    plan.detail['spec']=f'{num}{sm.group(2).upper()}'
                    plan.assumptions.append(f"capacity {plan.detail['spec']} was requested by the operator; policy still validates it against telemetry")
            return plan

        if re.search(r"older than|prior to|before|below|outdated", q) and re.search(r"macos|windows", q):
            m = re.search(r"(macos|windows)\b\D{0,40}?(\d+)", q)
            if not m:
                return Plan("unsupported", [], detail={"why": "Which OS version should I compare against (for example 'older than macOS 15')?"})
            if m:
                if re.search(r'(?:macos|windows).*?-\s*\d|(?:macos|windows).*?\d+\.\d+',q):
                    return Plan('unsupported',[],detail={'why':'OS comparisons require a positive whole major version; minor-version comparisons are not supported.'})
                prod = "macOS" if m.group(1) == "macos" else "Windows"
                return Plan("qa_os", [Step("count_os_below", {"product": prod, "major": int(m.group(2))}, "count devices below OS version")],
                            [f"'older' means major version < {m.group(2)} among {prod} devices; other OSes are excluded, not counted"])
        if re.search(r"trend|insight|pattern|drift|end[- ]of[- ]life|\beol\b|approaching|consistently|constrained|over time|wear", q):
            kinds = []
            if re.search(r"batter|end[- ]of[- ]life|\beol\b|wear", q): kinds.append("battery")
            if re.search(r"\bram\b|memory", q): kinds.append("ram")
            if re.search(r"storage",q) or (re.search(r"disk", q) and not _check_in(q)): kinds.append("disk")
            if re.search(r"drift|compliance", q) or _check_in(q): kinds.append("compliance_drift")
            kinds = kinds or ["disk", "ram", "battery", "compliance_drift"]
            return Plan("insights", [Step("detect_insights", {"kind": k, **({"check_id": _check_in(q)} if k=="compliance_drift" and _check_in(q) else {}), **({"form_factor":ff} if ff else {}), **({"device_id":devs[0]} if devs else {}), **({"severity":"high"} if k=="compliance_drift" and "high" in q else {})}, f"{k} trend detection") for k in kinds],
                        ["windows: the full 30-day telemetry history per device", "thresholds are the defaults in analytics.Thresholds"])
        if devs:
            return Plan("qa_device", [Step("get_device_history", {"device_id": devs[0]}, "device history")])
        if re.search(r"compliance|fail|check|encrypt|screen lock|policy", q):
            sev = "high" if re.search(r"high[- ]severity|severity high|\bhigh\b", q) else "medium" if re.search(r"medium[- ]severity", q) else None
            args = {k: v for k, v in {"severity": sev, "check_id": _check_in(q), "form_factor": ff}.items() if v}
            return Plan("qa_compliance", [Step("find_compliance_failures", args, "latest-snapshot failures")],
                        ["'failing' = status 'fail' in the device's most recent snapshot",
                         "laptop/desktop is inferred from model name (not a telemetry field)" if ff else "all form factors included"])
        if re.search(r"disk|storage|space", q):
            if re.search(r'\b(above|over|more than|greater than|at least|at most|no more than|between)\s+[-+]?\d|[<>]=|>\s*\d',q) or re.search(r'\d\s*(?:gb|gib|tb|tib)\b',q):
                return Plan('unsupported',[],detail={'why':'This read tool supports free-disk percentages below a threshold; greater-than and absolute-capacity filters are not supported.'})
            m = re.search(r"(?<![\w.])(-?(?:\d+(?:\.\d+)?|\.\d+))\s*(?:%|percent\b)", q)
            thr = float(m.group(1)) if m else 10.0
            return Plan("qa_disk", [Step("find_low_disk", {"threshold_pct": thr, **({"form_factor": ff} if ff else {})}, "latest free disk")],
                        [f"'low on disk space' = less than {thr:g}% free" + ("" if m else " (default; say if you want a different threshold)")])
        if re.search(r"overview|summary|fleet (size|status)|how many (devices|machines|laptops|employees)( are)?( there| do we have)?( in the fleet)?\s*\??\s*$", q):
            return Plan("qa_overview", [Step("fleet_overview", {"form_factor":ff} if ff else {}, "fleet summary")])
        return Plan("unsupported", [], detail={"why": "I can only answer fleet health, compliance and trend questions from telemetry, or propose "
                                                "tickets/upgrades/replacements/notifications for approval."})

    def _plan_action(self, act, q, devs, emps, ff) -> Plan:
        severity = "high" if re.search(r"\bhigh\b",q) else "medium" if re.search(r"\bmedium\b",q) else "low" if re.search(r"\blow[- ]severity\b",q) else None
        p = Plan(f"act_{act}", assumptions=["nothing is executed: every proposal waits for explicit human approval"], detail={"devs": devs, "emps": emps,"form_factor":ff,"severity":severity})
        p.detail['replacement_basis'] = 'battery' if re.search(r'batter|end[- ]of[- ]life|\beol\b',q) else 'disk' if re.search(r'disk|storage',q) else None
        if act == "ticket":
            cid = _check_in(q)
            if devs:
                p.detail["targets"] = [(d, cid) for d in devs]
                p.steps = [Step("get_device_history", {"device_id": d}, "confirm failures before proposing") for d in devs]
            else:
                p.steps = [Step("find_compliance_failures", {k: v for k, v in {"check_id": cid, "form_factor": ff,"severity":severity}.items() if v}, "select targets")]
        elif act == "upgrade":
            comp = "ram" if re.search(r"\bram\b|memory", q) else "disk"
            p.detail["component"] = comp
            if devs:
                p.steps = [Step("get_device_history", {"device_id": d}, "confirm constraint before proposing") for d in devs]
            else:
                p.steps = [Step("find_low_disk", {"threshold_pct": 10.0}, "select targets")] if comp == "disk" else [Step("detect_insights", {"kind": "ram"}, "select targets")]
        elif act == "replace":
            if devs:
                p.steps = [Step("get_device_history", {"device_id": d}, "confirm evidence before proposing") for d in devs]
            else:
                kinds = ['battery'] if re.search(r'batter|end[- ]of[- ]life|\beol\b',q) else ['disk'] if re.search(r'disk|storage',q) else ['battery','disk']
                p.steps = [Step('detect_insights',{'kind':kind},f'{kind} candidates') for kind in kinds]
        elif act == "notify":
            _all = [Step("find_compliance_failures", {k:v for k,v in {"form_factor":ff,"severity":severity,"check_id":_check_in(q)}.items() if v}, "open compliance findings"), Step("find_low_disk", {"threshold_pct": 10.0, **({"form_factor":ff} if ff else {})}, "open disk findings")]
            disk_t = bool(re.search(r"disk|storage|space", q)) and not _check_in(q)
            comp_t = bool(_check_in(q) or re.search(r"compliance|checks?|fail|policy|security", q))
            p.steps = [st for st, keep in zip(_all, (comp_t or not disk_t, disk_t or not comp_t)) if keep]
            if emps:
                p.detail["only_emps"] = emps
        return p

    # ---- second phase: turn observations into action proposals ----------------------
    def expand(self, plan: Plan, obs: list) -> list[Step]:
        act, out = plan.intent.removeprefix("act_"), []
        data = [(s, r) for s, r in obs if r.ok]
        if act == "ticket":
            for s, r in data:
                if s.tool == "find_compliance_failures":
                    for row in r.data["failures"]:
                        out.append(Step("open_remediation_ticket", {"device_id": row["device_id"], "check_id": row["check_id"],
                                        "note": f"Persistent '{row['check_id']}' failure ({row['severity']}) since {str(row['first_fail_ts'])[:10]}."}, "propose ticket"))
                elif s.tool == "get_device_history":
                    dev = r.data["device_id"]
                    cids = [c for d, c in plan.detail.get("targets", []) if d == dev and c] or \
                           [c for c, h in r.data["checks"].items() if h.get("latest") == "fail"]
                    for cid in cids:
                        out.append(Step("open_remediation_ticket", {"device_id": dev, "check_id": cid, "note": f"Persistent '{cid}' failure."}, "propose ticket"))
        elif act == "upgrade":
            comp = plan.detail["component"]
            for s, r in data:
                if s.tool == "find_low_disk":
                    for row in r.data["devices"]:
                        out.append(Step("create_upgrade_order", {"device_id": row["device_id"], "component": "disk", "spec": plan.detail.get("spec") or next_tier_spec(row["total_gb"])}, "propose upgrade"))
                elif s.tool == "detect_insights":
                    for f in r.data["findings"]:
                        if f["fields"]["tier"] == "consistent":
                            out.append(Step("create_upgrade_order", {"device_id": f["device_id"], "component": "ram", "spec": plan.detail.get("spec") or next_tier_spec(f["fields"]["ram_gb"])}, "propose upgrade"))
                elif s.tool == "get_device_history":
                    base = r.data["disk"]["total_gb"] if comp == "disk" else r.data["ram"]["ram_gb"]
                    out.append(Step("create_upgrade_order", {"device_id": r.data["device_id"], "component": comp, "spec": plan.detail.get("spec") or next_tier_spec(base)}, "propose upgrade"))
        elif act == "replace":
            seen = set()
            for s, r in data:
                if s.tool == "detect_insights":
                    for f in r.data["findings"]:
                        if f["severity"] in ("high", "critical") and f["device_id"] not in seen:
                            seen.add(f["device_id"])
                            out.append(Step("flag_device_for_replacement", {"device_id": f["device_id"], "reason": self._reason(r.data["kind"], f["fields"])}, "propose replacement"))
                elif s.tool == "get_device_history":
                    out.append(Step("flag_device_for_replacement", {"device_id": r.data["device_id"], "reason": "Operator-requested replacement review based on telemetry history."}, "propose replacement"))
        elif act == "notify":
            by_emp: dict = {}
            for s, r in data:
                rows = r.data.get("failures") or r.data.get("devices") or []
                for row in rows:
                    if plan.detail.get('devs') and row['device_id'] not in plan.detail['devs']:
                        continue
                    if plan.detail.get('form_factor') and self._view.latest(row['device_id']).form_factor != plan.detail['form_factor']:
                        continue
                    item = row.get("check_id") or "low disk space"
                    by_emp.setdefault(self._emp(row["device_id"], plan), []).append((row["hostname"], item))
            for emp, items in sorted(by_emp.items()):
                if plan.detail.get("only_emps") and emp not in plan.detail["only_emps"]:
                    continue
                txt = "; ".join(f"{h}: {i}" for h, i in sorted(set(items)))
                out.append(Step("notify_employee", {"employee_id": emp, "message": f"IT notice: open items on your device(s) - {txt}. Please contact IT support."}, "propose notification"))
        # Apply caller-requested target filters to every expanded proposal, including histories and insights.
        ff, severity = plan.detail.get('form_factor'), plan.detail.get('severity')
        def allowed(step):
            if step.tool=='flag_device_for_replacement' and plan.detail.get('replacement_basis'):
                step.args['basis']=plan.detail['replacement_basis']
            device = step.args.get('device_id')
            if device:
                latest = self._view.latest(device)
                if ff and latest.form_factor != ff:
                    return False
                if plan.detail.get('devs') and device not in plan.detail['devs']:
                    return False
                if step.tool == 'open_remediation_ticket' and severity:
                    return latest.checks.get(step.args['check_id'],{}).get('severity') == severity
            return True
        return [step for step in out if allowed(step)]

    _view = None

    def bind(self, view): self._view = view; return self
    def _emp(self, device_id, plan): return self._view.latest(device_id).employee_id

    @staticmethod
    def _reason(kind, f):
        if kind == "battery":
            return f"Battery 'Service Recommended' for {f['service_days']} days with capacity change {f['capacity_change_pct']}%."
        return f"Storage at {f['latest_free_pct']}% free with no recovery trend."

    def fallback(self, step: Step, result, obs) -> Step | None:
        """Replan on refusal: if policy says the proportionate action is different, try it once."""
        alts = result.data.get("alternatives", [])
        if step.hop >= 1 or not alts or step.args.get("component")=="ram":
            return None
        dev = step.args.get("device_id")
        if "flag_device_for_replacement" in alts and dev:
            return Step("flag_device_for_replacement", {"device_id": dev, "basis":"disk", "reason": f"{step.args.get('component', 'component')} constrained and cannot be upgraded in place."}, "fallback after refusal", hop=1)
        return None


class LLMPlanner(RulePlanner):
    """Optional. Needs ANTHROPIC_API_KEY. Untested in the offline eval environment
    (see README); every plan it returns is schema-validated and falls back to rules."""
    name = "llm"

    def __init__(self, model: str | None = None):
        self.model = model or os.environ.get("FLEET_COPILOT_MODEL")

    def plan(self, question: str, ctx: ToolContext) -> Plan:
        base = super().plan(question, ctx)
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key or not self.model or base.intent.startswith("act_") or base.intent == "unsupported":
            return base        # action flows keep the deterministic two-phase structure
        system = ("You plan tool calls for a fleet-telemetry assistant. Reply with JSON only: "
                  '{"steps":[{"tool":str,"args":object,"purpose":str}]}. Use only READ tools. '
                  "Never include tenant or company identifiers in args. Tool catalog: "
                  + json.dumps([t for t in catalog() if t["kind"] == "read"]))
        try:
            req = urllib.request.Request("https://api.anthropic.com/v1/messages", method="POST",
                headers={"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                data=json.dumps({"model": self.model, "max_tokens": 600, "system": system,
                                 "messages": [{"role": "user", "content": question}]}).encode())
            text = json.loads(urllib.request.urlopen(req, timeout=20).read())["content"][0]["text"]
            steps = []
            for s in json.loads(text[text.index("{"): text.rindex("}") + 1])["steps"][:6]:
                spec = REGISTRY[s["tool"]]
                clean, err = _validate(spec, s.get("args", {}))
                if spec.kind != "read" or err:
                    return base
                steps.append(Step(s["tool"], clean, s.get("purpose", "")))
            if steps:
                same_parameters = [(s.tool,s.args) for s in steps] == [(s.tool,s.args) for s in base.steps]
                assumptions = base.assumptions if same_parameters else ["The model-selected read parameters differ from deterministic defaults; actual filters and thresholds are shown in the results."]
                return Plan(base.intent, steps, assumptions + ["plan proposed by LLM, schema-validated"], base.detail)
        except Exception:  # noqa: BLE001
            pass
        return base
