"""Require a passing baseline, six caught injected faults, then restore the baseline report."""
import json
import os
import subprocess
import sys
from pathlib import Path
from eval.mutations import NAMES

clean = {k:v for k,v in os.environ.items() if k != 'FC_MUTATE'}
def run(env):
    return subprocess.run([sys.executable,'-m','eval.run_eval'],env=env,capture_output=True,text=True)
baseline=run(clean)
if baseline.returncode:
    print('Baseline failed; mutation results would be meaningless.\n'+baseline.stdout+baseline.stderr)
    raise SystemExit(1)
results=[]
for name in NAMES:
    result=run({**clean,'FC_MUTATE':name})
    caught=result.returncode==1
    results.append({'fault':name,'caught':caught,'exit_code':result.returncode})
    print(('CAUGHT ' if caught else 'SURVIVED/ERROR ')+name)
restored=run(clean)
Path('eval/mutation-report.json').write_text(json.dumps({'baseline_passed':True,'faults':results,'restored_baseline':restored.returncode==0},indent=2)+'\n')
raise SystemExit(0 if all(r['caught'] for r in results) and restored.returncode==0 else 1)
