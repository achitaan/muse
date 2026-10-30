"""Acceptance tests for the explicitly approved MoE limits, not kernel proofs."""

import ast
import hashlib
import json
import math

import torch

from build_tinyroute_bounded import OUT, OLD_SHA
from tinyroute_source import HERE


def main():
    raw=(HERE/'moe_tinyroute_bounded.py').read_bytes();old=(HERE/'moe_tinyroute.py').read_bytes()
    binding=json.loads((OUT/'source-binding.json').read_text(encoding='utf-8'))
    assert hashlib.sha256(raw).hexdigest()==binding['source_sha256']
    assert hashlib.sha256(old).hexdigest()==OLD_SHA
    tree=ast.parse(raw);before=ast.parse(old)
    for name in ('kernel_fn','_tr_launch','_ext','_make_scratch'):
        a=[ast.dump(n)for n in tree.body if isinstance(n,ast.FunctionDef)and n.name==name]
        b=[ast.dump(n)for n in before.body if isinstance(n,ast.FunctionDef)and n.name==name]
        assert a==b
    helpers=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name.startswith('_trb_')]
    env={'torch':torch,'HIDDEN':4096,'EMAX':3072}
    exec(compile(ast.Module(body=helpers,type_ignores=[]),'<approved-gate-only>','exec'),env)
    def snap(m=1):
        return dict(inter=torch.zeros(16,dtype=torch.bfloat16),ybuf=torch.zeros(16),
                    ss=torch.ones(8),out=torch.ones((m,4096),dtype=torch.bfloat16))
    def clone(d):return {k:v.clone()for k,v in d.items()}
    gate=env['_trb_gate'];a=snap();assert gate(a,clone(a))[0]
    checks=['identical']
    # Human-approved exact buffers distinguish signed zero despite equal floats.
    for key in ('inter','ybuf'):
        b=clone(a);b[key][0]=-0.;assert not gate(a,b)[0];checks.append(key+'_signed_zero')
    # Analytical powers of two straddle the1e-5 norm-sum relative limit.
    b=clone(a);b['ss'].add_(2.**-17);ok,r=gate(a,b)
    assert ok and math.isclose(r['ss']['relative_L2'],2.**-17,rel_tol=1e-12)
    b['ss'].copy_(a['ss']+2.**-16);assert not gate(a,b)[0];checks.extend(['ss_below','ss_above'])
    # BF16 nextafter(1) is1+2^-7. 64/4096 changed entries have global
    # relative error2^-10,below.001;256/4096 have2^-9,above.001.
    b=clone(a);b['out'][0,:64]=1.+2.**-7;ok,r=gate(a,b)
    assert ok and math.isclose(r['out']['relative_L2'],2.**-10,rel_tol=1e-12)
    b['out'][0,:256]=1.+2.**-7;assert not gate(a,b)[0];checks.extend(['out_below','out_above'])
    # Official-formula floor isolates a zero-norm row.127 other rows have
    # norm64,so its denominator is.1*(127*64/128)=6.35.
    a=snap(128);a['out'][0].zero_();b=clone(a);b['out'][0,0]=.03125;ok,r=gate(a,b)
    assert ok and math.isclose(r['out']['worst_row'],.03125/6.35,rel_tol=1e-12)
    b['out'][0,0]=.0625;ok,r=gate(a,b)
    assert not ok and r['out']['relative_L2']<.001 and r['out']['worst_row']>.005
    checks.extend(['row_floor_below','row_floor_above'])
    for key in ('inter','ybuf','ss','out'):
        b=clone(a);b[key].flatten()[0]=float('nan');assert not gate(a,b)[0];checks.append(key+'_nonfinite')
    b=clone(a);b['inter']=b['inter'].float();assert not gate(a,b)[0];checks.append('dtype')
    b=clone(a);b['out']=b['out'][:1];assert not gate(a,b)[0];checks.append('shape')
    a=dict(inter=torch.empty(0,dtype=torch.bfloat16),ybuf=torch.empty(0),ss=torch.empty(0),
           out=torch.zeros((1,4096),dtype=torch.bfloat16));assert gate(a,clone(a))[0];checks.append('empty_valid_prefix')
    # Source-verified completed-generation bank formula from diagnostic9dc89929.
    ib=torch.zeros(1032+3072,dtype=torch.int32);fb=torch.zeros(3072+4096);inter=torch.ones(4096,dtype=torch.bfloat16)
    fb[0]=17.;ib[1032:].view(torch.float32)[0]=29.;state={'scratch':[ib,fb,inter]}
    for generation in (1,2,3,4,901,902):
        ib[3]=generation;s=env['_trb_snapshot'](state,torch.zeros((1,4096),dtype=torch.bfloat16),1)
        assert s['generation']==generation and s['bank']==((generation-1)&1)
        assert s['ss'].item()==(17. if(generation-1)%2==0 else 29.)
        assert s['inter'].numel()==s['ybuf'].numel()==4096
    source=raw.decode();probe=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='_tr_probe')
    launch=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='_tr_launch')
    assert '_trb_gate'not in ast.unparse(launch)and '_trb_snapshot'not in ast.unparse(launch)
    assert 'No routing read is added to kernel_fn' in source
    record=dict(source_sha256=binding['source_sha256'],checks_passed=True,acceptance_cases=checks,
        generation_cases=[1,2,3,4,901,902],all_timed_definitions_unchanged=True,CUDA_unchanged=True,
        Decode_and_Attention_gate_unchanged=True,GPU_jobs=0,admitted_for_GPU=False,
        limitation='Gate formulas and source/snapshot ownership only; actual H200 worst-row bounds,selected new path and performance unverified.')
    (OUT/'cpu-audit.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8');print(json.dumps(record,indent=2))


if __name__=='__main__':main()
