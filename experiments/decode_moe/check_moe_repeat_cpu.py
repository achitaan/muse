"""CPU bounds/observation checks for one diagnostic; no native CUDA executed."""

import ast
import argparse
import base64
import contextlib
import hashlib
import io
import json
import random
import zlib
from pathlib import Path

import torch

from build_moe_repeat import OUT
from tinyroute_source import HERE, BASE_SHA


def functions(path, names, env):
    src = path.read_text()
    nodes = [n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(nodes) == len(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'diagnostic-CPU-only', 'exec'), env)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--compact',action='store_true'); args=parser.parse_args()
    out = OUT.parent / 'moe-repeat-v2' if args.compact else OUT
    runtime = 'moe_repeat_v2_runtime.py' if args.compact else 'moe_repeat_runtime.py'
    base = (HERE.parents[1] / 'best/moe_verified.py').read_bytes()
    blob = (HERE / ('moe_repeat_diag_v2.py' if args.compact else 'moe_repeat_diag.py')).read_bytes()
    assert hashlib.sha256(base).hexdigest() == BASE_SHA and blob.startswith(base)
    parent_tree, tree = ast.parse(base), ast.parse(blob)
    for name in ['kernel_fn','_ext','_make_scratch']:
        a = next(n for n in parent_tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        b = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
        assert len(b) == 1 and ast.dump(a, include_attributes=False) == ast.dump(b[0], include_attributes=False)
    H, E = 4096, 3072
    env = dict(torch=torch, HIDDEN=H, EMAX=E)
    functions(HERE / runtime, {'_mr_snapshot','_mr_difference'}, env)
    checked = []
    for n in [0, 1, 32, 256]:
        ib = torch.zeros(1032 + E, dtype=torch.int32)
        fb = torch.zeros(E + n*H, dtype=torch.float32)
        inter = torch.ones(n*H, dtype=torch.bfloat16)
        output = torch.zeros((max(1,n//8),H), dtype=torch.bfloat16)
        state = {'scratch':[ib,fb,inter]}
        fb[:E] = 17.; ib[1032:].view(torch.float32).fill_(29.)
        for gen in [1,2,3,4,901,902]:
            ib[3] = gen
            snap = env['_mr_snapshot'](state,output,n)
            expected = 17. if (gen-1)%2 == 0 else 29.
            assert snap['ss'].numel() == n and bool((snap['ss'] == expected).all())
            assert snap['inter'].numel() == snap['ybuf'].numel() == n*H
            same = env['_mr_difference'](snap,snap)
            assert all(d['byte_mismatches'] == 0 and d['finite'] for d in same.values())
        changed = {k:v.clone() if isinstance(v,torch.Tensor) else v for k,v in snap.items()}
        changed['out'].flatten()[0] = -0.
        diff = env['_mr_difference'](snap,changed)
        assert diff['out']['byte_mismatches'] > 0 and diff['out']['max_abs'] == 0.
        changed['out'].flatten()[0] = float('nan')
        diff = env['_mr_difference'](snap,changed)
        assert not diff['out']['finite']
        json.dumps(diff,allow_nan=False)
        checked.append(n)
    # Reuse the original synthetic fixture generator on CPU and independently
    # verify its expert-major valid prefix length and unique slots per row.
    fixture = dict(torch=torch)
    functions(HERE / 'tinyroute_runtime.py', {'_tr_mutate'}, fixture)
    cases = 0
    for m in [1,3,8,16,32]:
        d = dict(hidden_states=torch.empty((m,H),dtype=torch.bfloat16),
                 topk_ids=torch.empty((m,8),dtype=torch.int32),topk_weights=torch.empty((m,8)))
        for pattern in ['none','one','four','all','skew']:
            fixture['_tr_mutate'](d,{'lo':64,'cnt':32},pattern,0)
            rows = d['topk_ids'].tolist()
            valid = sorted((e,t,s) for t,row in enumerate(rows) for s,e in enumerate(row) if 64<=e<96)
            expected = sum({'none':0,'one':1,'four':4,'all':8}.get(pattern,1+t%3) for t in range(m))
            assert len(valid) == expected <= m*8 <= E
            assert all(len(set(row)) == 8 for row in rows)
            cases += 1
    record = dict(GPU_jobs=0, parent_prefix_exact=True, public_dispatch_ext_and_scratch_AST_unchanged=True,
                  valid_prefix_sizes=checked, bank_generations=[1,2,3,4,901,902],
                  signed_zero_detected=True, nonfinite_JSON_safe=True, fixture_cases=cases,
                  limitation='CPU observation/bounds only. Does not validate CUDA routing, atomic behavior, host compilation or snapshots on the judge.')
    if args.compact:
        functions(HERE / runtime, {'_mr_print_summary'}, env)
        # Stress the actual reporter, then retain only the judge-sized tail.
        rng = random.Random(381); reports=[]
        comparisons=['A_consecutive','C_consecutive','A_repeat','same_scratch_graph','independent_scratch',
                     'TR8_vs_same_scratch_parent','TR8_repeat']
        for m in [1,3,8,16,32]:
            for fixture in ['none','one','four','all','skew']:
                for seed in range(3):
                    fields={}
                    for comp in comparisons:
                        fields[comp]={}
                        for name in ['inter','ybuf','ss','out']:
                            fields[comp][name]=dict(byte_mismatches=rng.randrange(65536),finite=True,
                                max_abs=rng.random()*1e38,relative_L2=rng.random()*1e60,
                                samples=[[1,7,19],[rng.random()for i in range(3)],[rng.random()for i in range(3)]])
                    reports.append(dict(m=m,pattern=fixture,seed=seed,comparisons=fields,banks={'A':[0,1]}))
        captured=io.StringIO()
        with contextlib.redirect_stdout(captured):env['_mr_print_summary'](reports,['tiny','tiny2'],171.)
        text=captured.getvalue();tail=(text+'\nMR_DONE 171\n'+'post-prepare profiler\n'*60)[-16000:]
        lines=[s for s in tail.splitlines()if s.startswith('MR_SUMMARY_Z64 ')]
        assert len(lines)==1
        payload=lines[0].split(' ',1)[1];summary=json.loads(zlib.decompress(base64.b64decode(payload)))
        assert summary['cases']==75 and summary['variants']==['tiny','tiny2']
        assert all(n==15for n in summary['by_shape'].values())
        assert len(payload)<=9000 and len(summary['groups'])==35
        for m,c,stats in summary['groups']:
            for index,name in enumerate(summary['fields']):
                values=[r['comparisons'][c][name]for r in reports if r['m']==m]
                assert stats[index][0]==len(values)
                assert stats[index][1]==sum(v['byte_mismatches']>0for v in values)
                assert stats[index][3]==max(v['byte_mismatches']for v in values)
                assert stats[index][4]==max(v['max_abs']for v in values)
                assert stats[index][5]==max(v['relative_L2']for v in values)
        record['footer_stress']=dict(cases=75,groups=35,summary_z64_chars=len(payload),
            retained_inside_16000_char_tail_with_1260_char_profiler_suffix=True,all_aggregates_lossless=True)
        # FP64 diagnostic metrics remain finite for extreme finite FP32 data.
        for key in ['ss','ybuf']:
            snap[key].fill_(3e38)
        other={k:v.clone()if isinstance(v,torch.Tensor)else v for k,v in snap.items()}
        other['ss'].mul_(-1);other['ybuf'].mul_(-1)
        json.dumps(env['_mr_difference'](snap,other),allow_nan=False)
        record['extreme_finite_FP32_metric_JSON_safe']=True
    (out / 'cpu-audit.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))


if __name__ == '__main__': main()
