"""Only repair fixed-width false spelling; no GPU call or threshold changes."""
import ast
import contextlib
import copy
import io
import json
from pathlib import Path
import runpy
import sys

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'STATE.json').exists())
FOLDER=ROOT/'results/attention-skipscale-20261029'
original=ast.parse((FOLDER/'original-attention_skipscale_evidence.py').read_bytes())
current=ast.parse((ROOT/'scripts/attention_skipscale_evidence.py').read_bytes())
def control_assert(tree):
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='verify_trial')
    node=next(n for n in fn.body if isinstance(n,ast.Assert) and isinstance(n.msg,ast.Constant)
              and n.msg.value=='Nonflat L1 backend changed')
    return fn,node
old_fn,old=control_assert(original);new_fn,new=control_assert(current)
new_fn.body[new_fn.body.index(new)]=old
assert ast.dump(original,include_attributes=False)==ast.dump(current,include_attributes=False)
sys.path.insert(0,str(ROOT/'scripts'))
from attention_skipscale_evidence import verify_trial
with contextlib.redirect_stdout(io.StringIO()):
    scope=runpy.run_path(str(ROOT/'experiments/prepared/test_attention_skipscale_guard_cpu.py'))
    fixture=copy.deepcopy(scope['fixture'])
    for row in fixture['result']['records']:
        if row.get('event')=='profile' and row['index'] in (0,1,2,4):
            row['report']='void (anonymous namespace)::attn_kernel_skipscale<1, fals...'
    verify_trial(fixture)
rejected=[]
for label,spelling in (('changed_to_flat','true...'),('unrecognized_word','falsify...'),('missing_truncation_mark','fals')):
    bad=copy.deepcopy(fixture)
    next(r for r in bad['result']['records'] if r.get('event')=='profile' and r['index']==0)['report']='attn_kernel_skipscale<1,'+spelling
    try:
        with contextlib.redirect_stdout(io.StringIO()):verify_trial(bad)
    except AssertionError:rejected.append(label)
    else:raise AssertionError(('Bad template admitted',label))
report=dict(format_fixture_passed=True,rejected=rejected,original_13_guard_mutants_still_rejected=True,
            only_one_backend_assert_AST_changed=True,scientific_thresholds_unchanged=True,
            no_gpu_jobs=True,no_api_calls=True,fixtures_are_not_measurements=True)
path=FOLDER/'format-parser-cpu-audit.json';encoded=(json.dumps(report,indent=2)+'\n').encode()
assert not path.exists() or path.read_bytes()==encoded
path.write_bytes(encoded)
print(json.dumps(report,indent=2))
