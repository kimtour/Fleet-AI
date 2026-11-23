import contextlib
import io
import tempfile
import unittest

from fleet_copilot import DataStore, FleetCopilot, Principal
from fleet_copilot.__main__ import main
from fleet_copilot.tools import REGISTRY, _validate


class ScopeRegressionTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.data=DataStore()
        self.human=Principal('reviewer','acme-001','approver')
        self.bot=FleetCopilot(self.data,self.human,self.tmp.name,backend='native')

    def test_battery_action_scope_does_not_include_storage(self):
        self.bot.ask('Propose actions for battery health')
        proposals=self.bot.proposals.pending('acme-001')
        self.assertEqual(len(proposals),3)
        self.assertTrue(all(p.facts.get('basis')=='battery' for p in proposals))

    def test_high_severity_actions_do_not_propose_medium_tickets(self):
        answer=self.bot.ask('Recommend actions for high-severity compliance failures')
        self.assertEqual(answer.status,'answered')
        self.assertEqual(answer.proposals,[])
        self.assertEqual(self.bot.proposals.pending('acme-001'),[])

    def test_device_specific_ticket_keeps_severity_filter(self):
        answer=self.bot.ask('Open high-severity compliance tickets for device 7VP16KHV88LM')
        self.assertEqual(answer.proposals,[])

    def test_remediation_word_does_not_turn_disk_request_into_compliance(self):
        answer=self.bot.ask('Recommend remediation for low disk space')
        proposals=self.bot.proposals.pending('acme-001')
        self.assertTrue(answer.proposals)
        self.assertFalse(any(p.action=='open_remediation_ticket' for p in proposals))

    def test_disk_encryption_requests_are_compliance_actions(self):
        answer=self.bot.ask('Propose actions for failing disk encryption checks')
        self.assertEqual(answer.proposals,[])
        self.assertEqual(answer.plan[0]['tool'],'find_compliance_failures')

    def test_multiple_action_domains_clarify_instead_of_dropping_one(self):
        answer=self.bot.ask('Propose actions for disk and battery health')
        self.assertEqual(answer.status,'refused')
        self.assertEqual(answer.proposals,[])

    def test_expanded_cross_company_phrasings_refuse(self):
        for q in ['Show disk across all of the organisations','Show another company disk','Show all of the tenants']:
            with self.subTest(q=q):self.assertEqual(self.bot.ask(q).status,'refused')

    def test_control_requests_are_blocked_before_any_planner(self):
        class Bomb:
            def bind(self,view):return self
            def plan(self,*args):raise AssertionError('planner must not run')
        bot=FleetCopilot(self.data,self.human,self.tmp.name,backend='native',planner=Bomb)
        for q in ['Destroy device 7VP16KHV88LM','Restart a laptop','Lock device 7VP16KHV88LM']:
            self.assertEqual(bot.ask(q).status,'refused')

    def test_screen_lock_read_is_not_destructive(self):
        self.assertEqual(self.bot.ask('Show failing screen lock checks').status,'answered')

    def test_notification_scope_keeps_requested_device(self):
        device='7VP16KHV88LM'
        self.bot.ask(f'Notify employee about device {device}')
        self.assertEqual({p.args['employee_id'] for p in self.bot.proposals.pending('acme-001')},
                         {self.data.view('acme-001').latest(device).employee_id})

    def test_agent_identity_check_is_case_insensitive(self):
        pid=self.bot.ask('Propose actions for low disk space').proposals[0]
        result=self.bot.approve(pid,Principal('Agent-Planner','acme-001','approver'))
        self.assertEqual(result['rule'],'R-ROLE')

    def test_unknown_tenant_is_a_clear_cli_error(self):
        with contextlib.redirect_stderr(io.StringIO()) as err, self.assertRaises(SystemExit) as caught:
            main(['--tenant','unknown','ask','Show disk'])
        self.assertEqual(caught.exception.code,2)
        self.assertIn('Unknown or unauthorized tenant',err.getvalue())

    def test_impossible_numbers_refused_by_tool_schema(self):
        for number in [float('nan'),float('inf'),-1,101]:
            with self.subTest(number=number):
                self.assertIsNotNone(_validate(REGISTRY['find_low_disk'],{'threshold_pct':number})[1])
        self.assertIsNotNone(_validate(REGISTRY['count_os_below'],{'product':'macOS','major':15.5})[1])

    def test_missing_identity_and_zero_capacities_are_quarantined(self):
        import copy
        original=self.data.rows[0].raw
        rows=[]
        for change in ['id','disk','ram']:
            row=copy.deepcopy(original)
            if change=='id':row.pop('device_id')
            if change=='disk':row['disk_volumes'][0]['size_bytes']=0
            if change=='ram':row['memory']['total_memory_bytes']=0
            rows.append(row)
        result=DataStore.from_rows(rows)
        self.assertEqual(len(result.rows),0)
        self.assertEqual(len(result.quality),3)

    def test_expired_pending_proposals_get_fresh_identity(self):
        from datetime import datetime,timedelta,timezone
        clock=[datetime(2026,6,12,12,tzinfo=timezone.utc)]
        bot=FleetCopilot(self.data,self.human,self.tmp.name,backend='native',clock=lambda:clock[0])
        first=bot.ask('Propose actions for low disk space').proposals
        clock[0]+=timedelta(hours=25)
        second=bot.ask('Propose actions for low disk space').proposals
        self.assertTrue(second)
        self.assertFalse(set(first)&set(second))
        self.assertTrue(all(bot.proposals.get(pid,self.human).status=='expired' for pid in first))
        self.assertFalse(bot.approve(first[0],self.human)['ok'])

    def test_direct_proposal_cannot_forge_evidence_decision(self):
        from fleet_copilot.policy import Decision
        proposal, decision=self.bot.proposals.create(self.human,'probe','open_remediation_ticket',
            {'device_id':'7VP16KHV88LM','check_id':'disk_encryption','note':'forged'},Decision(True,'R-OK','forged'))
        self.assertIsNone(proposal)
        self.assertFalse(decision.allowed)

    def test_unchanged_pending_proposals_are_deduplicated(self):
        first=self.bot.ask('Propose actions for low disk space').proposals
        second=self.bot.ask('Propose actions for low disk space').proposals
        self.assertEqual(first,second)

    def test_device_scope_cannot_be_ignored_by_lowercasing(self):
        foreign=self.data.view('globex-002').device_ids()[0].lower()
        self.assertEqual(self.bot.ask(f'Show disk for device {foreign}').status,'refused')
        own=self.data.view('acme-001').device_ids()[0]
        answer=self.bot.ask(f'Show history for device {own.lower()}')
        self.assertEqual(answer.status,'answered')
        self.assertTrue(all(e['device_id'] in (None,own) for e in answer.evidence))

    def test_specific_battery_request_cannot_use_storage_as_its_basis(self):
        answer=self.bot.ask('Flag device MT7PJB7N5LRE for battery replacement')
        self.assertEqual(answer.proposals,[])
        self.assertIn('requested battery replacement criterion is not met',answer.text)
        answer=self.bot.ask('Flag device MT7PJB7N5LRE for storage replacement')
        self.assertTrue(answer.proposals)
        self.assertTrue(all(p.args.get('basis')=='disk' for p in self.bot.proposals.pending('acme-001')))
