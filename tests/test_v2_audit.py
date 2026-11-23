import subprocess
import sys
import tempfile
import unittest

from fleet_copilot import DataStore, FleetCopilot, Principal


class V2AuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.data=DataStore()
        self.bot=FleetCopilot(self.data,Principal('reviewer','acme-001','approver'),self.tmp.name,backend='native')

    def test_named_check_trends_exclude_unrequested_checks(self):
        answer=self.bot.ask('Show screen lock failures over time')
        self.assertIn('screen_lock',answer.text)
        self.assertNotIn('os_up_to_date',answer.text)
        self.assertNotIn('disk_encryption',answer.text)
        answer=self.bot.ask('Show disk encryption failures over time')
        self.assertNotIn('Storage:',answer.text)
        self.assertNotIn('screen_lock',answer.text)

    def test_device_and_form_factor_trend_filters_are_preserved(self):
        device='7VP16KHV88LM'
        answer=self.bot.ask(f'Show laptop battery trends for device {device}')
        self.assertEqual(answer.status,'answered')
        self.assertTrue(all(e['device_id'] in (None,device) for e in answer.evidence))

    def test_notifications_keep_named_check_scope(self):
        answer=self.bot.ask('Notify employees about disk encryption failures')
        self.assertEqual(answer.proposals,[])
        self.assertNotIn('low disk',answer.text)
        self.assertEqual(answer.plan[0]['args']['check_id'],'disk_encryption')

    def test_negative_requested_capacity_never_becomes_positive(self):
        answer=self.bot.ask('Order a -4TB disk for 8NM23J95R5I6')
        self.assertEqual(answer.status,'refused')
        self.assertEqual(answer.proposals,[])

    def test_from_to_capacity_uses_target(self):
        self.bot.ask('Upgrade disk for 8NM23J95R5I6 from 1TB to 4TB')
        self.assertEqual({p.args['spec'] for p in self.bot.proposals.pending('acme-001')},{'4TB'})

    def test_percent_word_and_fractional_thresholds_keep_values(self):
        for question,value in [('Show disks below 20 percent free',20),('Show disks below .5% free',.5)]:
            answer=self.bot.ask(question)
            self.assertEqual(answer.plan[0]['args']['threshold_pct'],value)

    def test_opposite_or_absolute_filters_are_explicitly_refused(self):
        for question in ['Show disk above 20% free','Show disk with more than 20% free','Show disk below 10GB free','Show disk with at most 20% free']:
            self.assertEqual(self.bot.ask(question).status,'refused',question)

    def test_unsupported_time_window_is_not_silently_full_history(self):
        self.assertEqual(self.bot.ask('Show disk trends for the last 7 days').status,'refused')

    def test_coordinated_negation_and_new_positive_clause(self):
        self.assertEqual(self.bot.ask('Show disk space and do not wipe or delete anything').status,'answered')
        self.assertEqual(self.bot.ask('Show disk space and do not delete then reboot devices').status,'refused')
        answer=self.bot.ask('Show low disk space and do not propose actions')
        self.assertEqual(answer.status,'answered')
        self.assertEqual(answer.proposals,[])

    def test_laptop_inventory_does_not_count_desktops(self):
        answer=self.bot.ask('How many laptops are there?')
        expected=sum(self.data.view('acme-001').latest(d).form_factor=='laptop' for d in self.data.view('acme-001').device_ids())
        self.assertIn(f'The fleet has {expected} devices',answer.text)
        self.assertEqual(answer.plan[0]['args']['form_factor'],'laptop')

    def test_unreported_check_is_unknown_not_healthy(self):
        from fleet_copilot.evidence import EvidenceLedger
        from fleet_copilot.policy import PolicyEngine
        from fleet_copilot.tools import ToolContext,call_tool
        from fleet_copilot.planner import Step
        from fleet_copilot.render import render
        view=self.data.view('acme-001')
        ctx=ToolContext(view,EvidenceLedger('acme-001'),self.bot.principal,'test',self.bot.audit,PolicyEngine(view),self.bot.proposals)
        step=Step('find_compliance_failures',{'check_id':'unreported_check'},'probe')
        result=call_tool(ctx,step.tool,step.args)
        claims,_=render([(step,result)],ctx)
        self.assertIn('cannot be determined',' '.join(c.text for c in claims))

    def test_all_failed_read_tools_mean_refusal(self):
        answer=self.bot.ask('Show disks below -5% free')
        self.assertEqual(answer.status,'refused')
        self.assertIn('policy range',answer.text)

    def test_shared_console_wrapper_handles_broken_pipe(self):
        for size in [10,1000000]:
            code=f"import sys; import fleet_copilot.__main__ as m; m.main=lambda: print('x'*{size}); sys.exit(m.cli())"
            process=subprocess.Popen([sys.executable,'-c',code],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            process.stdout.close()
            err=process.stderr.read().decode()
            process.stderr.close()
            self.assertEqual(process.wait(timeout=20),0,err)
            self.assertNotIn('Traceback',err)
        # Windows CRT reports a closed pipe as EINVAL rather than BrokenPipeError.
        for error, expected in [(22,0),(5,1)]:
            code=f"import sys; import fleet_copilot.__main__ as m; sys.platform='win32'; m.main=lambda: (_ for _ in ()).throw(OSError({error},'probe')); sys.exit(m.cli())"
            result=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True)
            self.assertEqual(result.returncode,expected,result.stderr)
        code="import sys; import fleet_copilot.__main__ as m; sys.platform='win32'; m.main=lambda: (_ for _ in ()).throw(OSError(22,'invalid file','invalid-path')); sys.exit(m.cli())"
        result=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0)

    def test_empty_trend_subset_is_citable_and_does_not_crash(self):
        view=self.data.view('acme-001')
        desktop=next(d for d in view.device_ids() if view.latest(d).form_factor=='desktop')
        answer=self.bot.ask(f'Show laptop compliance trends for device {desktop}')
        self.assertEqual(answer.status,'answered')
        self.assertIn('No matching',answer.text)

    def test_negative_and_minor_os_versions_require_clarification(self):
        for question in ['Show Windows older than -11','Show devices older than macOS 15.4']:
            self.assertEqual(self.bot.ask(question).status,'refused')
