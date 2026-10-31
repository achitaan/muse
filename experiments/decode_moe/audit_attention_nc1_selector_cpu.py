"""Exercise the retained Python launch path and its metadata-only guard."""
import ast
import hashlib
import itertools
import json
from pathlib import Path
from types import SimpleNamespace

from attention_nc1_contract import family, run_family

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/decode-moe-plan-20261030/attention-nc1'
SHA = '9e922c3555484081ca45a99a90284e26eccab181fb677700d8227a4b3402412f'


class Tensor:
    def __init__(self, shape=(1,)):
        self.shape = shape
        self.device = 'cuda:0'


class Recorder:
    def run(self, *args):
        self.args = args


def native_entry(args):
    # Independent transcription of the retained host selector, including ONE
    # then SPEC selection. Eligible flags0 needs no occupancy fallback model.
    B, L, _, _, P, window, _, ns, flat, nc = args[13:23]
    cs, flags = args[25], args[27]
    max_pages = args[5].shape[1]
    coop = bool((flags >> 1) & 1)
    if flat:
        return L, True, 0
    cl = cs > 1
    mode = 2 if cl else (3 if coop else 1)
    kmax = min(window + L - 1, max_pages * P) if window else max_pages * P
    if not cl and ns >= (kmax + 63) // 64:
        mode += 4
    if not cl and B == 1 and P == 1 and max_pages >= L:
        mode += 8
    return L, False, mode


def retained_launch_scope(raw=None):
    raw = (ROOT / 'best/attention_checked.py').read_bytes() if raw is None else raw
    assert hashlib.sha256(raw).hexdigest() == SHA
    tree = ast.parse(raw)
    scope = {}
    for n in tree.body:
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name):
            try:
                scope[n.targets[0].id] = ast.literal_eval(n.value)
            except (ValueError, TypeError):
                pass
    tuned = next(n for n in tree.body if isinstance(n, ast.Assign)
                 and getattr(n.targets[0], 'id', '') == '_TUNED')
    # literal_eval rejects the retained dict(...) constructors. Execute this
    # exact frozen metadata assignment instead of replacing it with an empty map.
    for call in (n for n in ast.walk(tuned.value) if isinstance(n, ast.Call)):
        assert isinstance(call.func, ast.Name) and call.func.id == 'dict' and not call.args
    exec(compile(ast.Module(body=[tuned], type_ignores=[]), '<retained-tuning>', 'exec'),
         {'__builtins__': {}, 'dict': dict}, scope)
    assert scope['_TUNED'][(1,1,False,1)] == dict(nc=8,coop=1,n_splits=252)
    names = ('_splits', '_copies', '_headroom', '_plan', '_launch')
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    scope.update(torch=SimpleNamespace(empty=lambda shape, **kw: Tensor(shape), bfloat16='bf16'),
                 _num_sms=lambda device: 132, _workspace=lambda *a, **kw: (None, None, None),
                 _inv_freq=lambda *a: None)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<retained-attention-launch>', 'exec'), scope)
    return scope


def main():
    scope = retained_launch_scope()
    cases, accepted = 0, {'flat_L1': 0, 'spec_one_L1': 0}
    for B, L, P, pages, window, pairs in itertools.product((1, 2, 16, 32), (1, 8),
            (1, 2, 8), (1, 31, 2048, 80001), (0, 2048), (1, 2)):
        cfg = dict(batch=B, q_len=L, num_q_heads=16 * pairs, num_kv_heads=2 * pairs,
                   head_dim=64, page_size=P, window=window, sm_scale=.125,
                   qk_scale_factor=1., rope_theta=10000.)
        data = dict(config=cfg, qkv=Tensor((B, L, 2304 * pairs)),
                    page_table=Tensor((B, pages)), seq_lens=Tensor(), cache_loc=Tensor(),
                    k_cache=Tensor(), v_cache=Tensor(), k_scale=Tensor(), v_scale=Tensor())
        for override in ({}, {'nc': 2}, {'nc': 8}, {'cs': 2}, {'coop': 1},
                         {'n_splits': 1}, {'flat': 1, 'nc': 1, 'coop': 0}):
            for pdl in (False, True):
                record = Recorder()
                scope['_launch'](data, _mod=record, pdl=pdl, **override)
                actual = record.args
                got = run_family(actual)
                native = native_entry(actual)
                if got is not None:
                    assert actual[22] == 1 and actual[27] == 0
                    assert native == ((1, True, 0) if got == 'flat_L1' else (1, False, 13))
                    accepted[got] += 1
                cases += 1
    # Boundary counterexamples: rejected changes are not silently absorbed by
    # the compiler specialization. These also kill omission/rounding mutants.
    base = dict(B=1, L=1, P=1, window=2048, max_pages=80001,
                n_splits=32, flat=0, nc=1, cs=0, flags=0)
    assert family(**base) == 'spec_one_L1'
    negative = [dict(nc=2), dict(nc=8), dict(L=8), dict(B=2), dict(P=2),
                dict(cs=2), dict(flags=2), dict(flags=1 << 16), dict(flags=1),
                dict(n_splits=31), dict(max_pages=0), dict(n_splits=0)]
    assert all(family(**dict(base, **change)) is None for change in negative)
    assert family(**dict(base, window=0, max_pages=65, n_splits=1)) is None
    assert family(**dict(base, window=0, max_pages=65, n_splits=2)) == 'spec_one_L1'
    assert family(**dict(base, flat=1, P=8, B=32)) == 'flat_L1'
    assert family(**dict(base, flat=1, cs=2)) is None
    assert family(**dict(base, window=-1)) is None
    report = dict(parent_sha256=SHA, actual_retained_launch_cases=cases, accepted=accepted,
                  boundary_counterexamples=len(negative) + 5,
                  exact_screened_entries_only=True, cooperative_and_PDL_excluded=True,
                  GPU_calls=0, admitted_for_GPU=False,
                  limitation='Metadata/selector evidence;no CUDA resource,cache-write,replay or numerical validation.')
    (OUT / 'selector-cpu-audit.json').write_bytes((json.dumps(report, indent=2) + '\n').encode())
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
