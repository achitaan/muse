"""CPU-only fixed-router safety, board-gate lineage and routing tests."""
import ast
import contextlib
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import threading
from types import SimpleNamespace

import torch
import verify_dense_consolidate as consolidation

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / 'candidates/dense_rtfix.py'


def fn(tree, name):
    return consolidation.function(tree, name)


def setup():
    tree = ast.parse(CANDIDATE.read_text())
    ns = dict(_dt_torch=torch, _dt_math=math, _dt_stats=statistics,
              _dt_time=SimpleNamespace(monotonic=lambda: 0.),
              _NG_TLS=threading.local())
    for node in tree.body:
        selected = isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id.startswith(('_DC_', '_RF_', '_DT_'))
            for target in node.targets for t in ast.walk(target))
        selected |= isinstance(node, ast.ClassDef) and node.name == '_DTBudget'
        selected |= isinstance(node, ast.FunctionDef) and node.name.startswith(('_rf_', '_dt_'))
        if selected:
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(CANDIDATE), 'exec'), ns)
    for name in ('prepare', 'kernel_fn'):
        exec(compile(ast.Module(body=[fn(tree, name)], type_ignores=[]), str(CANDIDATE), 'exec'), ns)
    return tree, ns


def lineage(tree):
    board = ast.parse((ROOT / 'board_best/dense_kbv2_scored_1.5886.py').read_text())
    base = ast.parse((ROOT / 'candidates/dense_consolidate.py').read_text())
    # All factory native strings, packing, flags, bounded plans and base routes
    # are byte-for-byte AST-identical to the caller's consolidated base.
    for name in ('_make_chenwainuo', '_make_vjeux', '_make_nb', '_make_ag', '_ng_empty'):
        assert ast.dump(fn(tree, name)) == ast.dump(fn(base, name)), name
    for node in board.body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith('_dt_'):
            assert ast.dump(fn(tree, node.name)) == ast.dump(node), node.name
    for name in ('prepare', 'kernel_fn'):
        old = fn(board, name)
        assert any(isinstance(n, ast.FunctionDef) and n.name == name and
                   ast.dump(n) == ast.dump(old) for n in tree.body), name
    assert not any(isinstance(n, ast.FunctionDef) and n.name.startswith('_dc_') for n in tree.body)
    assert not any(isinstance(n, ast.FunctionDef) and n.name == '_ng_draw' for n in tree.body)
    # Process/network imports and calls are forbidden; load_inline is inherited.
    forbidden = {'subprocess', 'multiprocessing', 'socket', 'requests', 'urllib', 'httpx'}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert not any(n.name.split('.')[0] in forbidden for n in node.names)
        if isinstance(node, ast.ImportFrom):
            assert node.module.split('.')[0] not in forbidden
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr not in {'Popen', 'system', 'fork', 'posix_spawn', 'execv', 'execve'}
    return dict(factory_and_flags_identical_to_consolidation=True,
                board_gate_and_prepare_verbatim=True, router_timing_gate_absent=True,
                forbidden_ast_clean=True)


class TorchCPU:
    cuda = SimpleNamespace(synchronize=lambda: None, get_device_capability=lambda d: (9, 0))

    def __getattr__(self, key):
        return getattr(torch, key)


def validator(ns):
    ns['_dt_torch'] = TorchCPU()
    ns['_V6_K'] = {'router': 64}
    # Analytical identity-column oracle: exactly representable BF16 inputs map
    # to the first 64 outputs unchanged, the other 192 outputs are exactly zero.
    w = torch.eye(256, 64)
    weights = {'router': w}
    calls = []
    def exact(data, state, arm):
        calls.append(data['x'].clone())
        return data['x'].float() @ w.t()
    ns['_rf_call'] = exact
    for m in (8, 16):
        calls.clear()
        assert ns['_rf_validate'](m, {}, weights, {}) == (0., 0.)
        assert len(calls) == 4 and not calls[-1].any()
        assert calls[2].count_nonzero().item() == m
        snapshot = [x.clone() for x in calls]
        calls.clear()
        ns['_rf_validate'](m, {}, weights, {})
        assert all(torch.equal(a, b) for a, b in zip(snapshot, calls))
    for kind in ('nan', 'inf', 'wrong', 'dtype', 'shape', 'zero-only'):
        def bad(data, state, arm):
            out = exact(data, state, arm)
            if kind in ('nan', 'inf'):
                out[0, 0] = float(kind)
            elif kind == 'wrong':
                out += 1
            elif kind == 'dtype':
                out = out.to(torch.bfloat16)
            elif kind == 'shape':
                out = out[:, :-1]
            elif not data['x'].any():
                out[0, 0] = 1
            return out
        ns['_rf_call'] = bad
        try:
            ns['_rf_validate'](8, {}, weights, {})
        except RuntimeError:
            pass
        else:
            raise AssertionError('accepted ' + kind)
    # Known global-vs-row arithmetic: one row with 8e-4 error has global 1e-4
    # over 64 rows but violates a 5*1e-4 per-row bound.
    ref = torch.ones(64, 2, dtype=torch.float64)
    out = ref.clone(); out[0] *= 1.0008
    err, row = ns['_dt_errors'](out, ref)
    assert abs(err - 1e-4) < 1e-12 and abs(row - 8e-4) < 1e-12
    assert not ns['_dt_close'](out, ref, 1e-4)
    return dict(identity_oracle=True, four_deterministic_batches_per_cell=True,
                nonfinite_wrong_dtype_shape_zero_fail_closed=True, row_floor_checked=True)


def board_gate(ns):
    class Graph:
        def replay(self):
            pass
    ns['_dt_torch'].cuda.CUDAGraph = Graph
    ns['_dt_torch'].cuda.graph = lambda graph: contextlib.nullcontext()
    ns['_V6_K'] = {'qkv_gate': 16}
    ns['_v6_n'] = lambda l, m: 16
    ns['_TABLE'] = {'qkv_gate:16': 'base'}
    ns['_ctx_parent_kernel_fn'] = lambda d, s: d['x'].clone()
    ns['_dt_call'] = lambda r, d, s: d['x'].clone()
    ns['_dt_equiv'] = lambda *args: True
    armkey = ('qkv_gate', 16, 't16s5curP-k7')
    state = {'dt_arms': {armkey: dict(kind='rq', p={'cnt': torch.zeros(1)})}}
    weights = {'o': torch.eye(16), 'qkv_gate': torch.eye(16)}
    for ratios, expected in (([(.988, .988), (.988, .988)], True),
                             ([(.989, .98)], False),
                             ([(.98, .989)], False),
                             ([(.98, .98), (.989, .98)], False)):
        times = iter(ratios)
        ns['_dt_abba'] = lambda *args: (1., 1., *next(times))
        got = ns['_dt_cell']('qkv_gate', 16, state, weights, None, None,
                             torch.Generator().manual_seed(7), float('inf'))
        assert (got == ('arm', armkey)) == expected
        assert list(times) == []
    return dict(board_inclusive_0_988_median_and_paired=True, fresh_confirmation_required=True)


def wiring(ns):
    ns['_dt_torch'] = TorchCPU()
    ns['_ctx_bind'] = lambda x: None
    ns['_rf_parent_prepare'] = lambda weights: {'dt': {('pooler', 32): 'CURP'}, 'other': 'kept'}
    packcalls, validations = [], []
    ns['_K'] = {'nb': {'rtpack': lambda w, s, wk: packcalls.append((s, wk)) or (s, wk)}}
    def validate(m, state, weights, arm):
        validations.append(m)
        if m == failure:
            raise RuntimeError('synthetic accuracy failure')
        return 1e-6, 2e-6
    ns['_rf_validate'] = validate
    weights = {'router': SimpleNamespace(is_cuda=True, device='cpu')}
    for failure in (None, 8, 16):
        validations.clear()
        with contextlib.redirect_stdout(io.StringIO()) as output:
            state = ns['prepare'](weights)
        assert validations == [8, 16]
        assert set(state['dense_rtfix']) == {('router', m) for m in (8, 16) if m != failure}
        assert state['dt'] == {('pooler', 32): 'CURP'} and state['other'] == 'kept'
        assert output.getvalue().count('DENSE_RTFIX cell=') == 2
        assert 'bound=5.000e-05' in output.getvalue()
        if failure:
            assert 'used=0 reason=RuntimeError:synthetic accuracy failure' in output.getvalue()
    assert packcalls == [(8, 8), (8, 16)] * 3
    validations.clear()
    with contextlib.redirect_stdout(io.StringIO()) as output:
        unsupported = ns['prepare']({'router': SimpleNamespace(is_cuda=False, device='cpu')})
    assert not unsupported['dense_rtfix'] and not validations
    assert output.getvalue().count('sm90 CUDA required') == 2
    failure = None
    with contextlib.redirect_stdout(io.StringIO()):
        state = ns['prepare'](weights)
    ns['_V6_K'] = {'router': 64}
    ns['_rf_parent_kernel_fn'] = lambda d, s: 'board'
    ns['_rf_call'] = lambda d, s, a: torch.empty((d['config']['m'], 256))
    for m in (8, 16):
        data = dict(x=torch.empty(m, 64, dtype=torch.bfloat16),
                    config=dict(layer='router', m=m, n=256, k=64))
        a, b = ns['kernel_fn'](data, state), ns['kernel_fn'](data, state)
        assert a.dtype == torch.float32 and a.data_ptr() != b.data_ptr()
        for key, value in (('n', 255), ('k', 63), ('m', 9), ('layer', 'pooler')):
            old = data['config'][key]; data['config'][key] = value
            assert ns['kernel_fn'](data, state) == 'board'
            data['config'][key] = old
        data['x'] = data['x'].float()
        assert ns['kernel_fn'](data, state) == 'board'
    return dict(two_defaults_no_timing=True, independent_failure_fallback=True,
                unsupported_fallback=True, board_selections_preserved=True,
                fresh_outputs_and_shape_guards=True)


def main():
    torch.set_num_threads(4)
    tree, ns = setup()
    report = dict(sha256=hashlib.sha256(CANDIDATE.read_bytes()).hexdigest(),
                  lineage=lineage(tree),
                  router_pack=consolidation.router_pack(tree),
                  bounded_rq=consolidation.bounded_rq(tree, ns),
                  numerical_safety=validator(ns), board_gate=board_gate(ns), wiring=wiring(ns),
                  gpu='NOT RUN: no CUDA/nvcc; compile, accuracy and prepare duration unverified')
    (ROOT / 'results/dense-rtfix-offline.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
