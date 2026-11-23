"""Fault injection. `FC_MUTATE=<name> python -m eval.run_eval` must FAIL for every name here;
`python -m eval.mutation_check` automates that. A suite that cannot fail proves nothing."""
from fleet_copilot import policy, tools
from fleet_copilot import data as data_mod, evidence

NAMES = ["no_tenant_isolation", "policy_allows_all", "auto_execute_on_propose", "fake_citations", "no_audit_chain", "tenant_args_accepted"]


def apply(name):
    if not name:
        return
    if name == "no_tenant_isolation":
        data_mod.TenantView.device_ids = lambda self: sorted(self._s._by_device)
        data_mod.TenantView.series = lambda self, d: self._s._by_device[d]
    elif name == "policy_allows_all":
        policy.PolicyEngine.check = lambda self, action, args: policy.Decision(True, "R-OK", "mutant", {})
    elif name == "auto_execute_on_propose":
        orig = policy.ProposalService.create
        def create(self, principal, session, action, args, decision):
            prop, d = orig(self, principal, session, action, args, decision)
            if prop:
                self._execute(prop, policy.Principal("agent", principal.tenant, "approver"))
            return prop, d
        policy.ProposalService.create = create
    elif name == "fake_citations":
        evidence.EvidenceLedger.add = lambda self, **kw: "E999"
    elif name == "no_audit_chain":
        orig_tail = evidence.AuditLog._tail
        evidence.AuditLog._tail = lambda self: (orig_tail(self)[0], evidence.AuditLog.GENESIS)   # records stop chaining
    elif name == "tenant_args_accepted":
        tools._validate = lambda spec, args: ({k: v for k, v in args.items() if k in spec.params}, None)
    else:
        raise SystemExit(f"unknown mutation {name}")
