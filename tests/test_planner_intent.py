import tempfile
import unittest

from fleet_copilot import DataStore, FleetCopilot, Principal


class PlannerIntentRegressions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.store = DataStore()

    def cop(self):
        return FleetCopilot(self.store, Principal("v", "acme-001", "viewer"), tempfile.mkdtemp(), backend="native")

    def test_negated_destructive_verbs_do_not_refuse_read_questions(self):
        a = self.cop().ask("Which devices have low disk space, and do not delete anything?")
        self.assertEqual(a.status, "answered")
        self.assertEqual(self.cop().ask("Wipe 8NM23J95R5I6").status, "refused")

    def test_unsupported_questions_do_not_fall_through_to_fleet_overview(self):
        self.assertEqual(self.cop().ask("Which devices were removed from the fleet?").status, "refused")
        self.assertEqual(self.cop().ask("Show fleet overview").status, "answered")

    def test_os_question_without_adjacent_number_is_parsed(self):
        a = self.cop().ask("How many devices are running Windows older than 11?")
        self.assertIn("Windows", a.text)
        self.assertNotIn("The fleet has", a.text)

    def test_named_check_over_time_returns_that_check_not_storage(self):
        a = self.cop().ask("Show screen lock failures over time")
        self.assertIn("screen_lock", a.text)
        self.assertNotIn("Storage:", a.text)

    def test_operator_requested_capacity_is_honoured_or_refused_never_silently_replaced(self):
        c = self.cop()
        c.ask("Order a 4TB disk for 8NM23J95R5I6")
        specs = {p.args["spec"] for p in c.proposals.pending("acme-001") if p.action == "create_upgrade_order"}
        self.assertEqual(specs, {"4TB"})
        a = c.ask("Order a 64TB disk for 8NM23J95R5I6")
        self.assertEqual(a.proposals, [])
        self.assertIn("R-SPEC", a.text)

    def test_negative_threshold_is_rejected(self):
        self.assertIn("policy range", self.cop().ask("Disk space below -5%").text)

    def test_notify_stays_on_the_requested_topic(self):
        c = self.cop()
        c.ask("Notify emp-acme-1005 about low disk")
        msg = [p.args["message"] for p in c.proposals.pending("acme-001") if p.action == "notify_employee"][0]
        self.assertIn("low disk", msg)
        self.assertNotIn("screen_lock", msg)


if __name__ == "__main__":
    unittest.main()
