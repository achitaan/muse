"""CPU host wiring and adversarial result guards, never a GPU measurement."""
import ast
import copy
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'STATE.json').exists())
sys.path.insert(0, str(ROOT / 'scripts'))
from decoder_moe_tail_evidence import TRIAL, CUDA, verify_report, verify_trial, parent_result, PARENT, full_result
from model_decoder_moe_tail import weight_addresses

WORK = ROOT / '.tools/decoder-moe-tail-20261029'


def synthetic_report():
    q = lambda big: [big,0,0,0,0,0,150,0,0,220000,132,1,1,288]
    rows = [dict(m=m,kind=k,dense_oracle=[.0001,.0002]) for m in (1,2,3,4,7,8)
            for k in ('normal','all_nonlocal','duplicate','outlier')]
    replay = [dict(cycle=c,index=i,m=m,max_error=[.0001,.0002]) for c in range(2)
              for i,m in enumerate((1,8,3,7,2,4,8,1))]
    paired = []
    for m in (1,8,3,7):
        for seed in (20261029,20261030):
            paired.append(dict(m=m,seed=seed,new_over_old=.94,saved_ns=6000.,
                sample_schema=['pair','first_arm','old_ns','new_ns'],
                samples=[[i,(i+seed)%2,100000.,94000.] for i in range(10)]))
    return dict(source_tag=CUDA[:12],cuda=CUDA,selected=True,weight_permutation_bitwise=True,
                resources=[dict(family=i,values=q(0)+q(1)) for i in range(2)],
                numerical=rows,comparison_maxima=[[.0001,.0002] for _ in range(4)],replay=replay,paired=paired)


def main():
    torch.set_num_threads(1)
    raw = (WORK / 'submission.py').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == TRIAL
    tree = ast.parse(raw)
    names = {'_dt_pack','_dt_call','_dt_eligible'}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    ns = dict(torch=torch)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<isolated CPU host functions>','exec'),ns)
    bits = np.random.default_rng(20261029).integers(-32768,32768,1024*4096,dtype=np.int16)
    weight = torch.from_numpy(bits).view(torch.bfloat16).reshape(1024,4096)
    actual = ns['_dt_pack'](weight).view(torch.int16).numpy()
    expected = bits[weight_addresses()].reshape(-1)
    assert np.array_equal(actual,expected)
    class FakeExt:
        def __init__(self): self.calls = 0
        def okc_run_tail(self,*args):
            assert len(args) == 10
            w,scratch,x,ids,tw,moe,lo,cnt,grid,tail = args
            m = x.shape[0]
            assert len(tail) == 9 and (lo,cnt,grid) == (0,32,132)
            assert ids.dtype == torch.int32 and tw.dtype == torch.float32
            assert moe.shape == (m,4096) and moe.dtype == torch.bfloat16
            assert [t.dtype for t in tail] == [torch.bfloat16,torch.bfloat16,torch.float32,
                torch.bfloat16,torch.bfloat16,torch.float32,torch.float32,torch.bfloat16,torch.float32]
            assert tail[1].shape == (m,1024) and tail[7].shape == tail[8].shape == (m,8192)
            assert len({t.data_ptr() for t in (moe,tail[1],tail[2],tail[7],tail[8])}) == 5
            self.calls += 1
    modules = [FakeExt(),FakeExt()]
    cache = {k:torch.ones(8192,dtype=torch.float32 if k in ('alpha','beta') else torch.bfloat16)
             for k in ('post_weight','pre_weight','alpha','beta')}
    state = dict(decoder_tail=dict(modules=modules,weight=weight.reshape(-1),scratch=[],selected=True),
                 native_moe=dict(w=[],lo=0,cnt=32,grid=132),sliced_ffn_boundary=cache)
    for m in range(1,9):
        x=torch.empty((m,4096),dtype=torch.bfloat16);stream=torch.empty((m,8192),dtype=torch.float32)
        ids=torch.empty((m,8),dtype=torch.int32);tw=torch.empty((m,8),dtype=torch.float32)
        assert ns['_dt_eligible'](x,stream,state)
        ns['_dt_call'](x,tw,ids,stream,state)
    assert [m.calls for m in modules] == [4,4]
    for m in (0,9,16,32,33,128,256,257):
        assert not ns['_dt_eligible'](torch.empty((m,4096)),torch.empty((m,8192)),state)
    state['decoder_tail']['selected']=False
    assert not ns['_dt_eligible'](x,stream,state)
    state['decoder_tail']=None
    assert not ns['_dt_eligible'](x,stream,state)
    binding=json.loads((WORK/'submission-binding.json').read_bytes())
    assert binding['launch_arguments']==10 and binding['tail_tensor_arguments']==9
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='okc_run_tail']
    assert len(calls)==1 and len(calls[0].args)==10 and not calls[0].keywords
    identity=[]
    for name in ('TASK.md','AGENTS.md','reference.py','checks.py','baseline.py','nvfp4.py'):
        a=(ROOT/'muse_decode_layer'/name).read_bytes()
        b=(ROOT/'.tools/decoder-moe-tail-identity-20261029'/name).read_bytes()
        identity.append(dict(file=name,pinned_sha256=hashlib.sha256(a).hexdigest(),
                             fresh_sha256=hashlib.sha256(b).hexdigest(),exact=a==b))
    assert all(r['exact'] for r in identity)
    (WORK/'evaluator-identity.json').write_bytes((json.dumps(dict(problem='muse_decode_layer',files=identity,GPU_jobs=0),indent=2)+'\n').encode())
    report = synthetic_report();verify_report(report)
    # Build a clearly labelled synthetic whole-layer result from the old schema.
    result = copy.deepcopy(parent_result(PARENT))
    body = result['result']
    for r in body['records']:
        if r.get('event') == 'benchmark' and r['index'] in (0,1,6,7):
            r['speedup'] *= 1.06
            r['median_ns'] = r['baseline_ns']/r['speedup']
        if r.get('event') == 'profile' and r['index'] in (0,1,6,7):
            r['report'] = 'SYNTHETIC GUARD FIXTURE ONLY: okc_tail::moe_fused'
    load={}
    for r in body['records']:
        if r.get('event')=='benchmark':
            a,b=load.setdefault(r['case']['load'],[0.,0.]);w=r['case']['weight']
            load[r['case']['load']]=[a+w*r['baseline_ns'],b+w*r['median_ns']]
    body['speedup']=math.prod(a/b for a,b in load.values())**(1/6)
    body['speedup_ci']=[body['speedup']*.985,body['speedup']*1.015]
    body['log_tail']='DECODER_TAIL '+json.dumps(report,separators=(',',':'))
    verify_trial(result,quiet=True)
    mutants=[]
    def reject(name,modify):
        broken=copy.deepcopy(result);modify(broken)
        try:verify_trial(broken,quiet=True)
        except (AssertionError,KeyError,ValueError,TypeError):mutants.append(name)
        else:raise AssertionError('guard mutation escaped: '+name)
    def mutate_report(name,modify):
        def apply(r):
            data=copy.deepcopy(report);modify(data)
            r['result']['log_tail']='DECODER_TAIL '+json.dumps(data,separators=(',',':'))
        reject(name,apply)
    for name,modify in (
        ('wrong_cuda',lambda r:r.update(cuda='wrong')),('empty_choice',lambda r:r.update(selected=False)),
        ('weight_mismatch',lambda r:r.update(weight_permutation_bitwise=False)),
        ('native_spill',lambda r:r['resources'][0]['values'].__setitem__(7,8)),
        ('bad_resource_api',lambda r:r['resources'][0]['values'].__setitem__(3,1)),
        ('no_residency',lambda r:r['resources'][0]['values'].__setitem__(11,0)),
        ('too_much_shared',lambda r:r['resources'][0]['values'].__setitem__(9,240000)),
        ('no_cooperative',lambda r:r['resources'][0]['values'].__setitem__(12,0)),
        ('missing_numeric',lambda r:r['numerical'].pop()),('bad_oracle',lambda r:r['numerical'][0].update(dense_oracle=[.01,.01])),
        ('bad_comparison',lambda r:r['comparison_maxima'].__setitem__(0,[.01,.01])),
        ('missing_replay',lambda r:r['replay'].pop()),('bad_replay',lambda r:r['replay'][0].update(max_error=[.01,.01])),
        ('missing_pair',lambda r:r['paired'].pop()),('bad_order',lambda r:r['paired'][0]['samples'][0].__setitem__(1,0)),
        ('fabricated_summary',lambda r:r['paired'][0].update(saved_ns=9000)),
        ('nonfinite_sample',lambda r:r['paired'][0]['samples'][0].__setitem__(2,float('nan'))),
    ):mutate_report(name,modify)
    reject('no_integrity',lambda r:r['result']['checks'].update(ok=False))
    reject('bad_CI',lambda r:r['result'].__setitem__('speedup_ci',[1.,1.]))
    reject('missing_profile',lambda r:r['result'].__setitem__('records',[q for q in r['result']['records'] if not(q.get('event')=='profile' and q['index']==0)]))
    reject('fallback_marker',lambda r:next(q for q in r['result']['records'] if q.get('event')=='profile' and q['index']==0).update(report='okc::moe_fused _sfn_kernel'))
    reject('wrong_weight',lambda r:next(q for q in r['result']['records'] if q.get('event')=='benchmark').get('case').update(weight=1))
    def change_load_gains(r, factors):
        totals={}
        originals={q['index']:q for q in parent_result(PARENT)['result']['records'] if q.get('event')=='benchmark'}
        for q in r['result']['records']:
            if q.get('event')!='benchmark':continue
            factor=factors(q['index'])
            q['speedup']=originals[q['index']]['speedup']*factor
            q['median_ns']=q['baseline_ns']/q['speedup']
            a,b=totals.setdefault(q['case']['load'],[0.,0.]);w=q['case']['weight']
            totals[q['case']['load']]=[a+w*q['baseline_ns'],b+w*q['median_ns']]
        r['result']['speedup']=math.prod(a/b for a,b in totals.values())**(1/6)
        r['result']['speedup_ci']=[r['result']['speedup']*.985,r['result']['speedup']*1.015]
    reject('insufficient_whole_gain',lambda r:change_load_gains(r,lambda i:1.001 if i in (0,1,6,7) else 1.))
    reject('control_drift',lambda r:change_load_gains(r,lambda i:1.06 if i in (0,1,6,7) else 1.10))
    reject('one_affected_load_loses',lambda r:change_load_gains(r,lambda i:.98 if i in (0,1) else 1.20 if i in (6,7) else 1.))
    host=dict(source_sha256=TRIAL,native_CUDA_sha256=CUDA,all_passed=True,CPU_only=True,GPU_jobs=0,
              mock_extension_calls=8,exact_original_BF16_values=4194304,shape_fallbacks=[0,9,16,32,33,128,256,257],
              distinct_live_output_allocations=True,compiled_Python=True,launch_arguments=10,tail_tensor_arguments=9,
              whole_result_positive_fixture_is_synthetic=True,guard_mutants_rejected=mutants,
              synthetic_compact_log_bytes=len(body['log_tail'].encode()),
              limits='CPU host/AST/schema tests; no native Torch/CUDA ABI, cooperative launch, GPU numerical or performance validation')
    (WORK/'host-CPU-admission.json').write_bytes((json.dumps(host,indent=2)+'\n').encode())
    print(json.dumps(host,indent=2))


if __name__=='__main__':main()
