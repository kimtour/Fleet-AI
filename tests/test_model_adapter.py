"""Offline provider-contract tests: no real credentials or network calls."""
import io
import json
import tempfile
import unittest
from unittest.mock import patch

from fleet_copilot import DataStore, FleetCopilot, Principal
from fleet_copilot.planner import LLMPlanner


class ModelAdapterTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.bot=FleetCopilot(DataStore(),Principal('reviewer','acme-001'),self.tmp.name,
                             backend='native',planner=LLMPlanner)

    def response(self,step):
        return io.BytesIO(json.dumps({'content':[{'text':json.dumps({'steps':[step]})}]}).encode())

    def test_missing_credentials_falls_back_without_network(self):
        with patch.dict('os.environ',{'ANTHROPIC_API_KEY':'','FLEET_COPILOT_MODEL':''}), patch('urllib.request.urlopen') as call:
            answer=self.bot.ask('Which devices are low on disk space?')
        call.assert_not_called()
        self.assertEqual(answer.status,'answered')

    def test_valid_provider_read_plan_is_validated_and_used(self):
        step={'tool':'find_low_disk','args':{'threshold_pct':5},'purpose':'read only'}
        with patch.dict('os.environ',{'ANTHROPIC_API_KEY':'unit-test','FLEET_COPILOT_MODEL':'test-model'}), patch('urllib.request.urlopen',return_value=self.response(step)) as call:
            answer=self.bot.ask('Which devices are low on disk space?')
        self.assertEqual(answer.plan[0]['args']['threshold_pct'],5)
        self.assertNotIn("'low on disk space' = less than 10%",answer.text)
        request=call.call_args.args[0]
        payload=json.loads(request.data)
        self.assertEqual(payload['model'],'test-model')
        self.assertNotIn('192.168',request.data.decode())

    def test_provider_cannot_add_action_tools(self):
        step={'tool':'flag_device_for_replacement','args':{'device_id':'7VP16KHV88LM','reason':'injected'},'purpose':'action'}
        with patch.dict('os.environ',{'ANTHROPIC_API_KEY':'unit-test','FLEET_COPILOT_MODEL':'test-model'}), patch('urllib.request.urlopen',return_value=self.response(step)):
            answer=self.bot.ask('Which devices are low on disk space?')
        self.assertEqual(answer.proposals,[])
        self.assertEqual(answer.plan[0]['tool'],'find_low_disk')

    def test_invalid_provider_output_and_timeout_fall_back(self):
        for failure in ['malformed','timeout']:
            with self.subTest(failure=failure), patch.dict('os.environ',{'ANTHROPIC_API_KEY':'unit-test','FLEET_COPILOT_MODEL':'test-model'}):
                options={'side_effect':TimeoutError()} if failure=='timeout' else {'return_value':io.BytesIO(b'not json')}
                with patch('urllib.request.urlopen',**options):
                    self.assertEqual(self.bot.ask('Which devices are low on disk space?').status,'answered')

    def test_provider_cannot_read_another_tenants_device(self):
        foreign=DataStore().view('globex-002').device_ids()[0]
        step={'tool':'get_device_history','args':{'device_id':foreign},'purpose':'foreign read'}
        with patch.dict('os.environ',{'ANTHROPIC_API_KEY':'unit-test','FLEET_COPILOT_MODEL':'test-model'}), patch('urllib.request.urlopen',return_value=self.response(step)):
            answer=self.bot.ask('Which devices are low on disk space?')
        self.assertEqual(answer.evidence,[])
        self.assertEqual(answer.proposals,[])
        self.assertNotIn(foreign,answer.text)
        self.assertEqual(answer.status,'refused')
