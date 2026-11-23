import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fleet_copilot import DataStore, FleetCopilot, Principal
from fleet_copilot.__main__ import main


class TransactionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.human = Principal('reviewer', 'acme-001', 'approver')
        self.bot = FleetCopilot(DataStore(),self.human,self.root,backend='native')
        self.pid = self.bot.ask('Propose disk upgrades for devices low on disk space').proposals[0]

    def test_process_crash_after_outbox_insert_rolls_back_everything(self):
        code = '''import os,sys
from fleet_copilot import DataStore,FleetCopilot,Principal
human=Principal('reviewer','acme-001','approver')
bot=FleetCopilot(DataStore(),human,sys.argv[1],backend='native')
original=bot.audit.append
def append(event,**kwargs):
    if event=='action_executed': os._exit(91)
    return original(event,**kwargs)
bot.audit.append=append
bot.approve(sys.argv[2],human)
'''
        result=subprocess.run([sys.executable,'-c',code,str(self.root),self.pid],capture_output=True,text=True)
        self.assertEqual(result.returncode,91,result.stderr)
        self.assertEqual(self.bot.proposals.get(self.pid,self.human).status,'pending')
        self.assertEqual(self.bot.proposals.executions(self.human.tenant),[])
        self.assertNotIn('proposal_approved',[r['event'] for r in self.bot.audit.records()])
        self.assertTrue(self.bot.audit.verify()[0])
        self.assertTrue(self.bot.approve(self.pid,self.human)['ok'])
        self.assertEqual(len(self.bot.proposals.executions(self.human.tenant)),1)

    def test_exception_after_outbox_insert_rolls_back(self):
        original=self.bot.audit.append
        def append(event,**kwargs):
            if event=='action_executed': raise RuntimeError('injected failure')
            return original(event,**kwargs)
        self.bot.audit.append=append
        with self.assertRaises(RuntimeError): self.bot.approve(self.pid,self.human)
        self.bot.audit.append=original
        self.assertEqual(self.bot.proposals.executions(self.human.tenant),[])
        self.assertEqual(self.bot.proposals.get(self.pid,self.human).status,'pending')
        self.assertTrue(self.bot.audit.verify()[0])

    def test_parallel_processes_execute_once(self):
        code='''import sys
from fleet_copilot import DataStore,FleetCopilot,Principal
p=Principal('reviewer','acme-001','approver')
b=FleetCopilot(DataStore(),p,sys.argv[1],backend='native')
print(b.approve(sys.argv[2],p)['ok'])
'''
        children=[subprocess.Popen([sys.executable,'-c',code,str(self.root),self.pid],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True) for _ in range(4)]
        outputs=[p.communicate(timeout=30) for p in children]
        self.assertTrue(all(p.returncode==0 for p in children),outputs)
        self.assertEqual(sum(out.strip()=='True' for out,err in outputs),1)
        self.assertEqual(len(self.bot.proposals.executions(self.human.tenant)),1)
        self.assertTrue(self.bot.audit.verify()[0])

    def test_expiry_tamper_is_refused(self):
        with self.bot.proposals.db.transaction() as db:
            p=json.loads(db.execute('SELECT payload FROM proposals WHERE id=?',(self.pid,)).fetchone()[0])
            p['created_at']=(datetime.now(timezone.utc)+timedelta(days=10)).isoformat()
            db.execute('UPDATE proposals SET payload=? WHERE id=?',(json.dumps(p),self.pid))
        self.assertEqual(self.bot.approve(self.pid,self.human)['rule'],'R-TAMPER')

    def test_cli_requires_exact_confirmation(self):
        with self.assertRaises(SystemExit) as result:
            main(['--state',str(self.root),'--role','approver','approve',self.pid,'--confirm','yes'])
        self.assertEqual(result.exception.code,2)
        self.assertEqual(self.bot.proposals.executions(self.human.tenant),[])

    def test_executed_action_cannot_be_reproposed_as_pending(self):
        self.assertTrue(self.bot.approve(self.pid,self.human)['ok'])
        answer=self.bot.ask('Propose disk upgrades for devices low on disk space')
        self.assertNotIn(self.pid,answer.proposals)
        self.assertEqual(self.bot.proposals.get(self.pid,self.human).status,'executed')
        self.assertEqual(len(self.bot.proposals.executions(self.human.tenant)),1)

    def test_changed_source_file_invalidates_cached_evidence(self):
        from fleet_copilot.data import DEFAULT_DATA
        path=self.root/'input.ndjson'
        path.write_bytes(DEFAULT_DATA.read_bytes())
        bot=FleetCopilot(DataStore(path),self.human,self.root/'changed',backend='native')
        pid=bot.ask('Propose disk upgrades for devices low on disk space').proposals[0]
        path.write_bytes(path.read_bytes()+b'\n')
        decision=bot.approve(pid,self.human)
        self.assertFalse(decision['ok'])
        self.assertEqual(decision['rule'],'R-EVIDENCE')
        self.assertEqual(bot.proposals.executions(self.human.tenant),[])

    def test_packaged_dataset_is_exact_source_copy(self):
        from fleet_copilot.data import DEFAULT_DATA
        source=Path('data/device-telemetry-dataset.ndjson')
        self.assertEqual(hashlib.sha256(source.read_bytes()).digest(),hashlib.sha256(DEFAULT_DATA.read_bytes()).digest())
