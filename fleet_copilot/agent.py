"""The Fleet Copilot agent: an explicit plan -> act -> reflect loop over a state graph.

    scope_guard -> plan -> act -> reflect -+-> act (next step / expanded step / fallback)
         |                                 +-> synthesize -> verify -> END
         +-> synthesize (refusal) ---------^

* scope_guard : refuses cross-tenant mentions, privilege-escalation/injection, out-of-chat approval
* plan        : planner emits typed tool steps (never data, never tenant ids)
* act         : runs ONE step through the validated, audited tool registry
* reflect     : decides what happens next from the observation: continue, expand a
                read result into action proposals, replan after a refusal (bounded), or stop
* synthesize  : renders claims from tool results (see render.py)
* verify      : drops any claim that lacks valid same-tenant evidence; blocks foreign ids
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from . import analytics as A
from .data import DataStore
from .evidence import AuditLog, EvidenceLedger
from .graph import END, build, resolve_backend
from .planner import unsupported_mutation, DEVICE_RE, LLMPlanner, Plan, RulePlanner
from .policy import PolicyEngine, Principal, ProposalService
from .render import Claim, render
from .tools import ToolContext, call_tool

MAX_STEPS = 40
ALL_TENANTS_RE = re.compile(r"\b(all|every|each|other|another|any|both)(?:\s+of)?(?:\s+the)?\s+(tenants?|compan(?:y|ies)|organi[sz]ations?|customers?|clients?)\b|"
                           r"\bacross\s+(all\s+)?(tenants?|compan(?:y|ies)|organi[sz]ations?)\b|\bcross[- ]tenant\b", re.I)
INJECTION_RE = re.compile(
    r"ignore (all |any |your |the )?(previous|prior|above|instructions)|disregard (the |your )?(rules|instructions|policy)|system prompt|"
    r"developer mode|jailbreak|disable (the )?guard|bypass|override (the )?(policy|guard)|without (human )?approval|skip (the )?approval|"
    r"auto[- ]?approve|approve (all|it|them|these|this|the|proposal|p-)|\bexecute (it|them|now|all)\b", re.I)


@dataclass
class Answer:
    status: str                       # answered | refused | partial | grounding_failed
    text: str
    claims: list[Claim] = field(default_factory=list)
    evidence: list[dict] = field(default_factory=list)
    proposals: list[str] = field(default_factory=list)
    plan: list[dict] = field(default_factory=list)
    dataset_sha256: str = ""
    session: str = ""


class FleetCopilot:
    def __init__(self, store: DataStore, principal: Principal, state_dir: Path | str = "state",
                 planner: Callable[[], RulePlanner] = RulePlanner, clock: Callable[[], datetime] | None = None,
                 backend: str = "auto", th: A.Thresholds = A.Thresholds(), audit_name: str = "audit.jsonl"):
        if principal.tenant not in store.tenants():
            raise ValueError('Unknown or unauthorized tenant')
        self.store, self.principal, self.th, self.backend = store, principal, th, resolve_backend(backend)
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.audit = AuditLog(Path(state_dir) / audit_name, self.clock)
        self.proposals = ProposalService(store, self.audit, state_dir, th)
        self.planner_factory = planner

    # ---------------------------------------------------------------- public API
    def ask(self, question: str) -> Answer:
        view = self.store.view(self.principal.tenant)
        sid = "S-" + hashlib.sha1(f"{self.principal.tenant}|{self.principal.admin_id}|{question}|{self.clock().isoformat()}".encode()).hexdigest()[:8]
        ctx = ToolContext(view, EvidenceLedger(self.principal.tenant), self.principal, sid, self.audit,
                          PolicyEngine(view, self.th), self.proposals, self.th)
        planner = self.planner_factory().bind(view)
        self.audit.append("question", tenant=self.principal.tenant, actor=self.principal.admin_id, session=sid, question=question)
        graph = self._graph(ctx, planner)
        state = {"question": question, "obs": [], "cursor": 0, "steps": [], "status": "ok", "expanded": False, "claims": [], "notes": []}
        final = (graph.invoke(state, {"recursion_limit": 500}) if self.backend == "langgraph" else graph.invoke(state))
        ans = final["answer"]
        ans.session = sid
        self.audit.append("answer", tenant=self.principal.tenant, actor="agent", session=sid, status=ans.status,
                          evidence=[e["id"] for e in ans.evidence], proposals=ans.proposals,
                          text_digest=hashlib.sha256(ans.text.encode()).hexdigest()[:16])
        return ans

    def approve(self, pid: str, approver: Principal) -> dict:
        return self.proposals.approve(pid, approver)

    def reject(self, pid: str, approver: Principal, reason: str = "") -> dict:
        return self.proposals.reject(pid, approver, reason)

    # ---------------------------------------------------------------- graph
    def _graph(self, ctx: ToolContext, planner: RulePlanner):
        me = self.principal

        def scope_guard(s):
            q = s["question"]
            if unsupported_mutation(q):
                return {'status':'refused','refusal':'This copilot cannot perform destructive or device-control operations. It can only answer from telemetry and propose review actions for human approval; nothing was changed.'}
            if ALL_TENANTS_RE.search(q):
                return {"status": "refused", "refusal": ("I can only access your own organisation's fleet. Requests that reference another "
                                                         "company's data, or all companies at once, are refused and have been logged.")}
            for t, aliases in self.store.tenant_aliases().items():
                if t != me.tenant and any(re.search(rf"\b{re.escape(a)}\b", q, re.I) for a in aliases):
                    return {"status": "refused", "refusal": ("I can only access your own organisation's fleet. Requests that reference another "
                                                             "company's data are refused and have been logged.")}
            if any(d.upper() not in ctx.view.device_ids() for d in DEVICE_RE.findall(q)):
                ctx.audit.append("guardrail_decision", tenant=me.tenant, actor="policy", session=ctx.session, rule="R-TENANT", allowed=False, reason="unknown device id in question")
                return {"status": "refused", "refusal": "That device was not found in your organisation."}
            if INJECTION_RE.search(q):
                return {"status": "refused", "refusal": ("I can't do that from chat. I never bypass guardrails, and approvals/executions are a separate "
                                                         "human step outside this conversation (`approve <proposal-id>`).")}
            return {}

        def plan(s):
            p = planner.plan(s["question"], ctx)
            # Validate model-proposed targets against the bound tenant before calling any tool.
            if any(step.args.get('device_id') is not None and step.args['device_id'] not in ctx.view.device_ids() for step in p.steps):
                ctx.audit.append('guardrail_decision',tenant=me.tenant,actor='policy',session=ctx.session,
                                 rule='R-TENANT',allowed=False,reason='planned device unavailable in this tenant')
                return {'status':'refused','refusal':'The planned target is unavailable in your organisation.'}
            ctx.audit.append("plan", tenant=me.tenant, actor="agent", session=ctx.session, intent=p.intent, planner=planner.name,
                             steps=[{"tool": x.tool, "args": x.args} for x in p.steps], assumptions=p.assumptions)
            if p.intent == "unsupported":
                return {"plan": p, "status": "refused", "refusal": p.detail["why"]}
            return {"plan": p, "steps": list(p.steps)}

        def act(s):
            step = s["steps"][s["cursor"]]
            res = call_tool(ctx, step.tool, step.args)
            return {"obs": s["obs"] + [(step, res)], "cursor": s["cursor"] + 1}

        def reflect(s):
            steps, cursor = list(s["steps"]), s["cursor"]
            step, res = s["obs"][-1]
            if not res.ok and step.tool in ("create_upgrade_order",) and (fb := planner.fallback(step, res, s["obs"])):
                steps.insert(cursor, fb)                       # replan: try the proportionate alternative once
            if cursor >= len(steps) and s["plan"].intent.startswith("act_") and not s["expanded"]:
                new = planner.expand(s["plan"], s["obs"])
                ctx.audit.append("replan", tenant=me.tenant, actor="agent", session=ctx.session, added=[{"tool": x.tool, "args": x.args} for x in new])
                steps, s = steps + new, {**s, "expanded": True}
            out = {"steps": steps, "expanded": s["expanded"]}
            if len(s["obs"]) >= MAX_STEPS:
                out["status"] = "partial"
                out["steps"] = steps[:cursor]
            return out

        def route_reflect(s):
            return "act" if s["cursor"] < len(s["steps"]) else "synthesize"

        def route_guard(s):
            return "synthesize" if s["status"] == "refused" else "plan"

        def route_plan(s):
            return "synthesize" if s["status"] == "refused" else ("act" if s["steps"] else "synthesize")

        def synthesize(s):
            if s["status"] == "refused":
                return {"claims": [Claim(s["refusal"], [], "system")]}
            claims, notes = render(s["obs"], ctx)
            all_failed=bool(s['obs']) and all(not result.ok for step,result in s['obs'])
            return {"claims": claims, "notes": notes, **({'status':'refused'} if all_failed else {})}

        def verify(s):
            ok, dropped = [], 0
            known = set(ctx.view.device_ids())
            for c in s["claims"]:
                if c.kind == "data":
                    bad = (not c.evidence or any((ctx.ledger.get(e) is None or ctx.ledger.get(e).tenant != me.tenant) for e in c.evidence)
                           or any(d not in known for d in DEVICE_RE.findall(c.text)))
                    if bad:
                        dropped += 1
                        continue
                ok.append(c)
            status = s["status"] if s["status"] in ("refused", "partial") else ("grounding_failed" if dropped else "answered")
            return {"claims": ok, "answer": self._assemble(s, ok, status, ctx, dropped)}

        nodes = dict(scope_guard=scope_guard, plan=plan, act=act, reflect=reflect, synthesize=synthesize, verify=verify)
        return build(nodes, {"act": "reflect", "synthesize": "verify", "verify": END},
                     {"scope_guard": route_guard, "plan": route_plan, "reflect": route_reflect}, "scope_guard", self.backend)

    # ---------------------------------------------------------------- output
    def _assemble(self, s, claims, status, ctx, dropped) -> Answer:
        lines, cited = [], []
        if status != "refused":
            lines.append(f"(tenant {self.principal.tenant}; telemetry as of {ctx.view.as_of.strftime('%Y-%m-%d %H:%M UTC')})")
        for c in claims:
            tag = " " + "".join(f"[{e}]" for e in c.evidence) if c.evidence else ""
            lines.append(c.text + tag)
            cited += [e for e in c.evidence if e not in cited]
        plan: Plan | None = s.get("plan")
        if plan and plan.assumptions and status != "refused":
            lines.append("Assumptions: " + "; ".join(plan.assumptions) + ".")
        for n in dict.fromkeys(s.get("notes", [])):
            lines.append("Note: " + n)
        if dropped:
            lines.append(f"Warning: {dropped} statement(s) were withheld because they could not be tied to evidence.")
        if status == "partial":
            lines.append(f"Warning: stopped at the {MAX_STEPS}-step budget; results are partial.")
        props = [o[1].data["proposal_id"] for o in s["obs"] if o[1].ok and "proposal_id" in o[1].data]
        if props:
            lines.append(f"{len(set(props))} proposal(s) are PENDING human approval; nothing has been executed.")
        ev = [e.__dict__ for e in (ctx.ledger.get(i) for i in cited) if e]
        if ev:
            lines.append("Evidence: " + "; ".join(f"[{e['id']}] {e['device_id'] or 'fleet'} {e['kind']} @ {e['ts']} ({e['source']})" for e in ev))
        return Answer(status, "\n".join(lines), claims, ev, sorted(set(props)),
                      [{"tool": x.tool, "args": x.args} for x in s.get("steps", [])],
                      dataset_sha256=self.store.source_hash())
