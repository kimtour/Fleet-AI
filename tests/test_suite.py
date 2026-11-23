import unittest

from fleet_copilot import analytics as A
from fleet_copilot.data import DataStore, parse_ts


class TestAnalytics(unittest.TestCase):
    def test_ols_perfect_line(self):
        s, _, r2 = A.ols([0, 1, 2, 3], [10, 8, 6, 4])
        self.assertAlmostEqual(s, -2.0)
        self.assertAlmostEqual(r2, 1.0)

    def test_disk_trend_plateau_means_effectively_full(self):
        rows = [{"device_id": "D1", "company_id": "t", "employee_id": "e", "collected_at": f"2026-05-{i + 1:02d}T00:00:00Z",
                 "os": {"hostname": "h", "product_name": "macOS", "product_version": "14.5", "architecture": "arm64"},
                 "device_identity": {"model_name": "MacBook Pro"}, "memory": {"total_memory_bytes": 100, "used_memory_bytes": 50},
                 "disk_volumes": [{"size_bytes": 1000, "available_bytes": a}], "installed_software": [], "compliance_results": []}
                for i, a in enumerate([200, 150, 100, 60, 40, 20, 20, 20, 20])]
        t = A.disk_trend(DataStore.from_rows(rows).view("t").series("D1"))
        self.assertEqual(t["days_to_full"], 0.0)
        self.assertGreaterEqual(t["plateau_snapshots"], 4)

    def test_malformed_and_duplicate_rows_are_quarantined(self):
        good = {"device_id": "D1", "company_id": "t", "employee_id": "e", "collected_at": "2026-05-01T00:00:00Z",
                "os": {"hostname": "h", "product_name": "macOS", "product_version": "14.5", "architecture": "arm64"},
                "device_identity": {"model_name": "MacBook Pro"}, "memory": {"total_memory_bytes": 100, "used_memory_bytes": 50},
                "disk_volumes": [{"size_bytes": 1000, "available_bytes": 500}], "installed_software": [], "compliance_results": []}
        bad = {**good, "collected_at": "2026-05-02T00:00:00Z", "disk_volumes": [{"size_bytes": 1000, "available_bytes": 5000}]}
        st = DataStore.from_rows([good, good, bad])
        self.assertEqual(len(st.rows), 1)
        self.assertEqual(len(st.quality), 2)


class TestLangGraphSchema(unittest.TestCase):
    """Guard declared channels in both verified graph backends.
    LangGraph drops state keys that are not declared in the schema."""

    def test_state_schema_declares_every_key_a_run_produces(self):
        import tempfile
        from fleet_copilot.agent import FleetCopilot
        from fleet_copilot.graph import AgentState
        from fleet_copilot.policy import Principal

        store = DataStore("data/device-telemetry-dataset.ndjson")
        cop = FleetCopilot(store, Principal("tester", "acme-001", "viewer"), tempfile.mkdtemp())
        seen = set()
        orig = cop._graph

        def spy(ctx, planner):
            g = orig(ctx, planner)
            inner = g.invoke

            def invoke(state, *a, **kw):      # LangGraph passes a config argument
                out = inner(state, *a, **kw)
                seen.update(out)
                return out
            g.invoke = invoke
            return g

        cop._graph = spy
        for q in ("Which devices at Acme are low on disk space?", "Propose disk upgrades for devices low on disk space",
                  "Show Globex devices low on disk"):
            cop.ask(q)
        self.assertTrue(seen)
        self.assertLessEqual(seen, set(AgentState.__annotations__), seen - set(AgentState.__annotations__))

    def test_employee_count_note_present(self):
        store = DataStore("data/device-telemetry-dataset.ndjson")
        notes = store.view("acme-001").quality_notes()
        self.assertTrue(any("distinct employee id" in n for n in notes))


class TestTenantLeakAndConcurrency(unittest.TestCase):
    def _row(self, did, tenant, ts="2026-05-01T00:00:00Z", free=500):
        return {"device_id": did, "company_id": tenant, "employee_id": "e", "collected_at": ts,
                "os": {"hostname": "h", "product_name": "macOS", "product_version": "14.5", "architecture": "arm64"},
                "device_identity": {"model_name": "MacBook Pro"}, "memory": {"total_memory_bytes": 100, "used_memory_bytes": 50},
                "disk_volumes": [{"size_bytes": 1000, "available_bytes": free}], "installed_software": [], "compliance_results": []}

    def test_ingestion_notes_never_leak_other_tenants(self):
        rows = [self._row("GLOBEX-DEV", "globex-002"), self._row("GLOBEX-DEV", "acme-001"),   # id collision
                self._row("GLOBEX-DUP", "globex-002"), self._row("GLOBEX-DUP", "globex-002"),  # duplicate
                self._row("GLOBEX-BAD", "globex-002", ts="2026-05-02T00:00:00Z", free=9999),   # malformed
                self._row("ACME-1", "acme-001")]
        st = DataStore.from_rows(rows)
        blob = " ".join(st.view("acme-001").quality_notes())
        for forbidden in ("GLOBEX", "globex"):
            self.assertNotIn(forbidden, blob)

    def test_stale_instance_cannot_reject_executed_proposal_or_fork_audit(self):
        import tempfile
        from fleet_copilot.agent import FleetCopilot
        from fleet_copilot.policy import Principal

        d = tempfile.mkdtemp()
        store = DataStore("data/device-telemetry-dataset.ndjson")
        human = Principal("sam", "acme-001", "approver")
        a = FleetCopilot(store, human, d)
        ans = a.ask("Propose disk upgrades for devices low on disk space")
        pid = ans.proposals[0]
        stale = FleetCopilot(store, human, d)          # loaded while the proposal is still pending
        stale.proposals.pending("acme-001")
        self.assertTrue(a.approve(pid, human)["ok"])
        res = stale.reject(pid, human, "too late")
        self.assertFalse(res["ok"])
        self.assertEqual(a.proposals.get(pid, human).status, "executed")
        self.assertTrue(a.audit.verify()[0], a.audit.verify())
        self.assertTrue(stale.audit.verify()[0])
        stale.audit.append("probe", tenant="acme-001", actor="t", session="s")      # stale writer must not fork the chain
        self.assertTrue(a.audit.verify()[0])

    def test_parallel_approvals_execute_once(self):
        import tempfile, threading, json as _j
        from pathlib import Path
        from fleet_copilot.agent import FleetCopilot
        from fleet_copilot.policy import Principal

        d = tempfile.mkdtemp()
        store = DataStore("data/device-telemetry-dataset.ndjson")
        human = Principal("sam", "acme-001", "approver")
        pid = FleetCopilot(store, human, d).ask("Propose disk upgrades for devices low on disk space").proposals[0]
        results = []
        ts = [threading.Thread(target=lambda: results.append(FleetCopilot(store, human, d).approve(pid, human))) for _ in range(4)]
        [t.start() for t in ts]; [t.join() for t in ts]
        self.assertEqual(sum(r["ok"] for r in results), 1)
        self.assertEqual(len(FleetCopilot(store, human, d).proposals.executions(human.tenant)), 1)
        self.assertTrue(FleetCopilot(store, human, d).audit.verify()[0])


class TestProbeRegressions(unittest.TestCase):
    def _cop(self):
        import tempfile
        from fleet_copilot.agent import FleetCopilot
        from fleet_copilot.policy import Principal
        return FleetCopilot(DataStore(), Principal("v", "acme-001", "viewer"), tempfile.mkdtemp(), backend="native")

    def test_destructive_requests_are_refused_not_downgraded_to_a_read(self):
        c = self._cop()
        for q in ("Delete device 8NM23J95R5I6", "Wipe 8NM23J95R5I6 remotely", "Decommission MT7PJB7N5LRE", "Reboot all devices"):
            a = c.ask(q)
            self.assertEqual(a.status, "refused", q)
            self.assertIn("nothing was changed" if "Reboot" not in q else "only answer", a.text)
            self.assertEqual(a.proposals, [])

    def test_screen_lock_questions_are_not_mistaken_for_mutations(self):
        a = self._cop().ask("Show screen lock failures over time")
        self.assertNotEqual(a.status, "refused")

    def test_all_tenant_requests_are_refused(self):
        c = self._cop()
        for q in ("show all tenants devices low on disk", "compare with other companies", "across all organisations list devices"):
            self.assertEqual(c.ask(q).status, "refused", q)

    def test_generic_propose_actions_creates_proposals(self):
        c = self._cop()
        for q, n in (("Propose actions for low disk space", 2), ("Recommend actions for batteries approaching end-of-life", 1),
                     ("Propose actions for consistently constrained RAM", 1)):
            a = c.ask(q)
            self.assertGreaterEqual(len(a.proposals), 1, (q, a.text[:200]))
        self.assertEqual(c.ask("Propose actions").status, "refused")      # ambiguous: asks which area


class TestExpiry(unittest.TestCase):
    def test_pending_proposal_expires_after_24h(self):
        import tempfile
        from datetime import datetime, timedelta, timezone
        from fleet_copilot.agent import FleetCopilot
        from fleet_copilot.policy import Principal

        now = [datetime(2026, 6, 12, 12, tzinfo=timezone.utc)]
        store = DataStore("data/device-telemetry-dataset.ndjson")
        cop = FleetCopilot(store, Principal("sam", "acme-001", "approver"), tempfile.mkdtemp(), clock=lambda: now[0])
        ans = cop.ask("Propose disk upgrades for devices low on disk space")
        self.assertTrue(ans.proposals)
        now[0] += timedelta(hours=25)
        res = cop.approve(ans.proposals[0], Principal("sam", "acme-001", "approver"))
        self.assertFalse(res["ok"])
        self.assertEqual(res["rule"], "R-EXPIRED")


class TestEvalSuite(unittest.TestCase):
    def test_full_eval_passes(self):
        from eval.run_eval import main
        self.assertEqual(main(), 0)


if __name__ == "__main__":
    unittest.main()
