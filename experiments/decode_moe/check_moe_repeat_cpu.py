"""CPU bounds/observation checks for one diagnostic; no native CUDA executed."""

import ast
import hashlib
import json
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
    base = (HERE.parents[1] / 'best/moe_verified.py').read_bytes()
    blob = (HERE / 'moe_repeat_diag.py').read_bytes()
    assert hashlib.sha256(base).hexdigest() == BASE_SHA and blob.startswith(base)
    parent_tree, tree = ast.parse(base), ast.parse(blob)
    for name in ['kernel_fn','_ext','_make_scratch']:
        a = next(n for n in parent_tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        b = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
        assert len(b) == 1 and ast.dump(a, include_attributes=False) == ast.dump(b[0], include_attributes=False)
    H, E = 4096, 3072
    env = dict(torch=torch, HIDDEN=H, EMAX=E)
    functions(HERE / 'moe_repeat_runtime.py', {'_mr_snapshot','_mr_difference'}, env)
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
    (OUT / 'cpu-audit.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))


if __name__ == '__main__': main()
