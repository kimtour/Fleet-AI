"""CLI. Identity is simulated with flags; in production the principal comes from the IdP token."""
from __future__ import annotations

import argparse
import json
import sys

from . import FleetCopilot, DataStore, Principal
from .planner import LLMPlanner, RulePlanner
from .tools import catalog


def _mk(a):
    store = DataStore(a.data)
    p = Principal(a.admin, a.tenant, a.role)
    planner = LLMPlanner if a.planner == "llm" else RulePlanner
    return store, p, FleetCopilot(store, p, a.state, planner=planner, backend=a.backend)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="fleet_copilot")
    ap.add_argument("--data", default=None)
    ap.add_argument("--state", default="state")
    ap.add_argument("--tenant", default="acme-001")
    ap.add_argument("--admin", default="admin@acme")
    ap.add_argument("--role", default="viewer", choices=["viewer", "approver"])
    ap.add_argument("--planner", default="rules", choices=["rules", "llm"])
    ap.add_argument("--backend", default="auto", choices=["auto", "native", "langgraph"])
    sub = ap.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("ask"); q.add_argument("question"); q.add_argument("--json", action="store_true")
    sub.add_parser("chat")
    for n in ("approve", "reject"):
        s = sub.add_parser(n); s.add_argument("proposal_id")
        if n == "approve": s.add_argument("--confirm", required=True)
    sub.add_parser("proposals")
    sub.add_parser("audit-verify")
    sub.add_parser("tools")
    a = ap.parse_args(argv)
    if a.data is None:
        from .data import DEFAULT_DATA
        a.data = str(DEFAULT_DATA)
    if a.cmd == "tools":
        print(json.dumps(catalog(), indent=2)); return 0
    try:
        store, me, bot = _mk(a)
    except (ValueError, OSError) as exc:
        ap.error(str(exc))
    if a.cmd == "ask":
        ans = bot.ask(a.question)
        print(json.dumps({"status": ans.status, "text": ans.text, "proposals": ans.proposals, "evidence": ans.evidence, "dataset_sha256": ans.dataset_sha256}, indent=2) if a.json else ans.text)
        return 0 if ans.status == "answered" else 2
    elif a.cmd == "chat":
        print(f"Fleet Copilot - tenant {me.tenant}, user {me.admin_id} ({me.role}). /proposals /reject <id> /quit  (approve with: approve <id> --confirm \"APPROVE <id>\")")
        for line in sys.stdin if not sys.stdin.isatty() else iter(lambda: input("> "), None):
            line = line.strip()
            if line in ("/quit", "exit"): break
            if line == "/proposals": _show(bot, me)
            elif line.startswith("/approve "): print("Use the dedicated approve command with --confirm, after reviewing the proposal.")
            elif line.startswith("/reject "): print(bot.reject(line.split()[1], me))
            elif line: print(bot.ask(line).text)
    elif a.cmd == "approve":
        if a.confirm != "APPROVE " + a.proposal_id:
            ap.error("--confirm must be exactly APPROVE followed by the proposal ID")
        result = bot.approve(a.proposal_id, me)
        print(json.dumps(result, indent=2))
        return 0 if result["ok"] else 2
    elif a.cmd == "reject":
        result = bot.reject(a.proposal_id, me)
        print(json.dumps(result, indent=2))
        return 0 if result["ok"] else 2
    elif a.cmd == "proposals":
        _show(bot, me)
    elif a.cmd == "audit-verify":
        ok, msg = bot.audit.verify(); print(("OK" if ok else "FAIL"), msg, f"({sum(r['tenant'] == me.tenant for r in bot.audit.records())} tenant records)"); return 0 if ok else 1
    return 0


def _show(bot, me):
    for p in bot.proposals.pending(me.tenant):
        print(f"{p.id}  {p.action}  {json.dumps(p.args)}  [{p.rule}]")


def cli():
    """Shared module/console entry point; closed output pipes exit quietly."""
    try:
        result = main()
        sys.stdout.flush()  # Catch small buffered writes before interpreter shutdown.
        return result
    except BrokenPipeError:
        import os
        try:
            fd=os.open(os.devnull,os.O_WRONLY)
            try:
                os.dup2(fd,sys.stdout.fileno())
            finally:
                os.close(fd)
        except (OSError,ValueError):
            pass
        return 0


if __name__ == '__main__':
    sys.exit(cli())
