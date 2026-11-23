"""Answer synthesis.

Answers are *rendered from tool results*, not free-written: every number and name in a
data claim is pulled from a structured result that carries evidence ids. That makes
"unsupported conclusions" structurally hard rather than something we hope a model avoids.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Claim:
    text: str
    evidence: list[str]
    kind: str = "data"          # data | system (refusals, guardrail notes: no telemetry evidence needed)


def _gib(g):
    return f"{g / 1024:g} TiB" if g >= 1024 and g % 1024 == 0 else f"{g:g} GiB"


def render(obs: list, ctx) -> tuple[list[Claim], list[str]]:
    claims: list[Claim] = []
    notes: list[str] = []
    for step, r in obs:
        if not r.ok and r.tool in _ACTIONS:
            c, n = _action(step, r, ctx)
            claims += c
            continue
        if not r.ok:
            host = _host(ctx, step.args.get("device_id"))
            claims.append(Claim(f"Could not {step.tool.replace('_', ' ')}{' for ' + host if host else ''}: {r.error} (rule {r.rule})", [], "system"))
            continue
        fn = globals().get(f"_r_{r.tool}")
        if fn:
            c, n = fn(step, r, ctx)
            claims += c
            notes += n
    return claims, notes


def _host(ctx, dev):
    try:
        return f"{ctx.view.latest(dev).hostname} ({dev})" if dev else None
    except Exception:  # noqa: BLE001
        return dev


def _r_fleet_overview(step, r, ctx):
    o = r.data["overview"]
    c = [Claim(f"The fleet has {o['devices']} devices ({o['employees']} employees) with {o['snapshots']} snapshots from {o['first_ts']} to {o['last_ts']}. "
               f"By OS: {o['by_os']}; by form factor (inferred): {o['by_form_factor']}.", [r.data["ev"]])]
    return c, list(r.data["quality_notes"])


def _r_find_low_disk(step, r, ctx):
    d, thr, rows = r.data, r.data["threshold_pct"], r.data["devices"]
    ff = step.args.get("form_factor")
    scope = f"{ff}s " if ff else ""
    cov = d["cov"]
    if not rows:
        return [Claim(f"No {scope}devices are below {thr:g}% free disk space in their latest snapshot.", [cov])], []
    out = [Claim(f"{len(rows)} {scope}device(s) have less than {thr:g}% free disk space in their latest snapshot:", [cov])]
    for x in rows:
        out.append(Claim(f"- {x['hostname']} ({x['device_id']}): {x['free_pct']}% free ({_gib(x['free_gb'])} of {_gib(x['total_gb'])})", [x["ev"]]))
    return out, []


def _r_find_compliance_failures(step, r, ctx):
    d, f, rows, cov = r.data, r.data["filters"], r.data["failures"], r.data["cov"]
    subject = f"{f['form_factor']}s" if f["form_factor"] else "devices"
    kind = f"{f['severity']}-severity " if f["severity"] else ""
    kind += f"'{f['check_id']}' check" if f["check_id"] else "compliance check"
    if not rows:
        evd = ctx.ledger.get(cov).fields["checks_present"]
        if f["severity"]:
            evd = [c for c in evd if c.endswith(f":{f['severity']}")]
        if f['check_id']:
            evd = [c for c in evd if c.startswith(f['check_id']+':')]
        if not evd:
            return [Claim('No matching compliance check observations are reported; pass/fail cannot be determined.',[cov])], []
        return [Claim(f"No {subject} are failing any {kind}: all {d['devices_in_scope']} in-scope {subject} pass every matching check in their latest snapshot "
                      f"(matching checks reported: {', '.join(evd) or 'none'}).", [cov])], []
    out = [Claim(f"{len({x['device_id'] for x in rows})} of {d['devices_in_scope']} {subject} fail a {kind} in the latest snapshot "
                 f"({len(rows)} failing check result(s)):", [cov])]
    for x in rows:
        out.append(Claim(f"- {x['hostname']} ({x['device_id']}): {x['check_id']} [{x['severity']}] failing for {x['streak']} consecutive snapshot(s)"
                         f"{', first seen ' + x['first_fail_ts'][:10] if x['first_fail_ts'] else ''}", [x["ev"]]))
    return out, []


def _r_count_os_below(step, r, ctx):
    d = r.data
    n, m = len(d["older"]), len(d["older"]) + len(d["not_older"])
    out = [Claim(f"{n} of {m} {d['product']} devices run a {d['product']} version older than {d['major']}.", [d["cov"]])]
    for x in d["older"]:
        out.append(Claim(f"- {x['hostname']} ({x['device_id']}): {x['os']}", [x["ev"]]))
    notes = [f"{d['other_os_excluded']} non-{d['product']} device(s) were excluded from this count."] if d["other_os_excluded"] else []
    return out, notes


def _r_get_device_history(step, r, ctx):
    d, ev, i = r.data, r.data["ev"], r.data["identity"]
    k, rm, b = d["disk"], d["ram"], d["battery"]
    out = [Claim(f"{i['hostname']} ({d['device_id']}) - {i['model']}, {i['os']}, owner {i['employee_id']}.", [ev["disk"]]),
           Claim(f"Disk: free space {k['first_free_pct']}% -> {k['latest_free_pct']}% over {k['n']} snapshots ({k['slope_pp_per_day']} pp/day)"
                 + (f"; projected full in ~{k['days_to_full']} days" if k["days_to_full"] else ""), [ev["disk"]]),
           Claim(f"RAM: mean {rm['mean_pct']}% used, p95 {rm['p95_pct']}%; above {85}% in {rm['snaps_above']}/{rm['n']} snapshots", [ev["ram"]])]
    if b.get("has_battery_data"):
        out.append(Claim(f"Battery: condition '{b['condition_latest_known']}', capacity {b['first_capacity']} -> {b['last_capacity']} ({b['capacity_change_pct']}%)", [ev["battery"]]))
    else:
        out.append(Claim("Battery: no battery readings in any snapshot.", [ev["battery"]]))
    fails = {c: h for c, h in d["checks"].items() if h.get("latest") == "fail"}
    out.append(Claim("Compliance: " + ("; ".join(f"{c} failing for {h['current_fail_streak']} snapshot(s)" for c, h in sorted(fails.items())) or "all checks passing"), [ev["checks"]]))
    return out, []


def _r_detect_insights(step, r, ctx):
    d, kind, fs = r.data, r.data["kind"], r.data["findings"]
    out, notes = [], []
    if kind == "disk":
        out.append(Claim(f"Storage: {len(fs)} device(s) under 20% free ({sum(f['severity'] == 'critical' for f in fs)} critical <=5%).", [d["cov"]]))
        for f in fs:
            x = f["fields"]
            why = (f"volume has sat at ~{x['latest_free_pct']}% for {x['plateau_snapshots']} snapshots (effectively full)" if x["days_to_full"] == 0 else
                   f"fell {x['first_free_pct']}% -> {x['latest_free_pct']}% ({x['slope_pp_per_day']} pp/day), full in ~{x['days_to_full']} days at this rate" if x["days_to_full"] else
                   "no sustained decline")
            out.append(Claim(f"- [{f['severity']}] {f['title']} ({f['device_id']}): {x['latest_free_pct']}% free; {why}; below 10% in {x['snaps_below_low']}/{x['n']} snapshots", [f["ev"]]))
    elif kind == "ram":
        out.append(Claim(f"Memory: {sum(f['fields']['tier'] == 'consistent' for f in fs)} device(s) consistently constrained (>85% used in >=70% of snapshots), "
                         f"{sum(f['fields']['tier'] == 'intermittent' for f in fs)} intermittently.", [d["cov"]]))
        for f in fs:
            x = f["fields"]
            out.append(Claim(f"- [{f['severity']}] {f['title']} ({f['device_id']}): {x['tier']}; >85% in {x['snaps_above']}/{x['n']} snapshots, mean {x['mean_pct']}%, p95 {x['p95_pct']}% of {x['ram_gb']} GB", [f["ev"]]))
        if fs:
            notes.append("'used memory' is as reported by the agent; its definition (e.g. whether cache counts) may differ by OS, so treat RAM findings as pressure indicators.")
    elif kind == "battery":
        out.append(Claim(f"Batteries: {len(fs)} device(s) show service-recommended condition or declining capacity.", [d["cov"]]))
        for f in fs:
            x = f["fields"]
            svc = f"'Service Recommended' since {x['service_since']} ({x['service_days']} days)" if x["service_since"] else f"condition {x['condition_latest_known']}"
            out.append(Claim(f"- [{f['severity']}] {f['title']} ({f['device_id']}): {svc}; full-charge capacity {x['first_capacity']} -> {x['last_capacity']} ({x['capacity_change_pct']}% over the window); cycles {x['cycles_first']} -> {x['cycles_last']}", [f["ev"]]))
        if fs:
            notes.append("Design capacity is not in telemetry, so absolute battery health % and remaining life are NOT computed; change is relative to each device's first reading.")
        for g in d.get("gaps", []):
            notes.append(f"{g['device_id']}: battery block missing in {g['missing']}/{g['of']} snapshots (agent data gap); battery findings use the last known reading.")
    elif kind == "compliance_drift":
        if not fs:
            out.append(Claim('No matching compliance trend observations are reported.',[d['cov']]))
        for f in fs:
            w = f["weekly"]
            series = " -> ".join(f"{x['failing']}/{x['devices_seen']}" for x in w)
            out.append(Claim(f"{f['check_id']}: devices failing per week (oldest -> latest): {series}.", [f["ev"]]))
            for x in f["regressions"]:
                out.append(Claim(f"- {x['hostname']} ({x['device_id']}) regressed pass->fail at {x['regressed_at']} and has failed {x['streak']} snapshot(s) since", [x["ev"]]))
            if f["chronic"]:
                out.append(Claim(f"- {len(f['chronic'])} device(s) failed {f['check_id']} in every snapshot (chronic, no drift): {', '.join(f['chronic'])}", [f["ev"]]))
    return out, notes


def _action(step, r, ctx):
    a = r.data["args"]
    host = _host(ctx, a.get("device_id")) or a.get("employee_id")
    if not r.ok:
        alt = f" Suggested alternative: {', '.join(r.data.get('alternatives', []))}." if r.data.get("alternatives") else ""
        return [Claim(f"REFUSED {r.tool} for {host}: {r.error} (rule {r.rule}).{alt}", [], "system")], []
    tag = " (already pending)" if r.data.get("duplicate") else ""
    detail = {"create_upgrade_order": f"{a.get('component')} -> {a.get('spec')}", "open_remediation_ticket": a.get("check_id", ""),
              "flag_device_for_replacement": "replacement review", "notify_employee": "notification"}[r.tool]
    return [Claim(f"PROPOSED {r.tool} for {host} [{detail}] as {r.data['proposal_id']}{tag}; evidence re-derived by policy from source telemetry.", r.evidence)], []


_ACTIONS = ("create_upgrade_order", "open_remediation_ticket", "flag_device_for_replacement", "notify_employee")
for _t in _ACTIONS:
    globals()[f"_r_{_t}"] = _action
