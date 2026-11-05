"""CPU-only consolidation/gate regression tests; never import the submission."""
from __future__ import annotations

import ast
import contextlib
import copy
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import threading
from types import SimpleNamespace

import torch

import verify_rotcodes_null as legacy

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / 'candidates/dense_consolidate.py'


def function(tree, name):
    return [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name][-1]


def setup():
    tree = ast.parse(CANDIDATE.read_text())
    ns = dict(_ng_contextlib=contextlib, _ng_threading=threading, _ng_torch=torch,
              _ng_math=math, _ng_stats=statistics, _dt_torch=torch,
              _dt_time=SimpleNamespace(monotonic=lambda: 0.))
    for node in tree.body:
        selected = isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id.startswith(('_NG_', '_DC_'))
            for target in node.targets for t in ast.walk(target))
        selected |= isinstance(node, ast.ClassDef) and node.name in ('_NGBindingError', '_DTBudget')
        selected |= isinstance(node, ast.FunctionDef) and node.name.startswith(('_ng_', '_dc_', '_dt_'))
        if selected:
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(CANDIDATE), 'exec'), ns)
    for name in ('prepare', 'kernel_fn'):
        exec(compile(ast.Module(body=[function(tree, name)], type_ignores=[]), str(CANDIDATE), 'exec'), ns)
    return tree, ns


def lineage(tree):
    board = ast.parse((ROOT / 'board_best/dense_kbv2_scored_1.5886.py').read_text())
    null = ast.parse(legacy.CANDIDATE.read_text())
    normalized = copy.deepcopy(tree)
    count = [0]

    class Unbind(ast.NodeTransformer):
        def visit_Call(self, n):
            self.generic_visit(n)
            if isinstance(n.func, ast.Name) and n.func.id == '_ng_empty':
                n.func = ast.parse('torch.empty').body[0].value
                count[0] += 1
            return n

    for name in ('_make_chenwainuo', '_make_vjeux', '_make_nb', '_make_ag'):
        original, current = function(board, name), Unbind().visit(function(normalized, name))
        if name in ('_make_vjeux', '_make_ag'):
            assert ast.dump(original) == ast.dump(current), name
        elif name == '_make_chenwainuo':
            # All native CUDA strings, packing, base preparation, math and bakes
            # precede the bounded-DT additions. Only the optional compiler helper
            # is excluded, because it now accepts an explicit C++20 build flag.
            stop = next(i for i, n in enumerate(original.body) if isinstance(n, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == '_DTRQ_NAMES' for t in n.targets))
            a = [n for n in original.body[:stop] if not isinstance(n, ast.FunctionDef) or n.name != '_rqext']
            b = [n for n in current.body[:stop] if not isinstance(n, ast.FunctionDef) or n.name != '_rqext']
            assert ast.dump(ast.Module(body=a, type_ignores=[])) == ast.dump(ast.Module(body=b, type_ignores=[]))
        else:
            def string(fn, symbol):
                return next(n.value.value for n in fn.body if isinstance(n, ast.Assign)
                            and any(isinstance(t, ast.Name) and t.id == symbol for t in n.targets))
            old = string(original, '_RT_CUDA')
            new = string(current, '_RT_CUDA')
            assert new == old.replace('RSM(8,8) RSM(16,8) RSM(16,4)',
                                      'RSM(8,8) RSM(16,8) RSM(16,4) RSM(8,16)')
            assert string(current, '_OL_CUDA') == string(original, '_OL_CUDA')
            assert ast.dump(function(current, '_rt_pack')) == ast.dump(function(original, '_rt_pack'))
    assert count[0] == 21, count  # the 22nd original site belongs to global _v6_call
    for name in ('_ng_empty', '_ng_borrow', '_ng_aligned_empty', '_ng_call_into',
                 '_ng_capture_orders', '_ng_schedule', '_ng_summarize', '_ng_phase', '_ng_draw'):
        assert ast.dump(function(tree, name)) == ast.dump(function(null, name)), name
    # No old ABBA admission or previous rotated-code preparation is reachable.
    assert not any(isinstance(n, ast.FunctionDef) and n.name in ('_dt_abba', '_dt_tune', '_r2_start', '_rc_start')
                   for n in tree.body)
    assert CANDIDATE.read_text().startswith('# okc: problem=muse_bf16_proj\n')
    assert b'\r' not in CANDIDATE.read_bytes()
    return dict(base_native_math_unchanged=True, router_dispatch_addition='RSM(8,16)',
                controlled_primitives_identical_to_fa02f58=True)


def metric(value, paired=None):
    paired = value if paired is None else paired
    return dict(median=value, paired=paired,
                panels=[dict(median=value, paired=paired, order=o) for o in ('AB', 'BA', 'BA', 'AB')],
                orders={o: dict(median=value, paired=paired) for o in ('AB', 'BA')})


def record(nulls, candidate=.95, paired=None):
    pre, post = metric(1.), metric(1.)
    for p, value in zip(pre['panels'] + post['panels'], nulls):
        p.update(median=value, paired=value)
    return dict(pre=pre, candidate=metric(candidate, paired), post=post)


def policy(ns):
    model = ns['_dc_null_model']
    gate = ns['_dc_gate']
    # Independent known arithmetic: these 8 values have median 1 and MAD .01;
    # the requested 3*MAD bound is .03, not the old fixed +/- .005.
    r = record([.98, .99, 1., 1.01, 1.02, .99, 1., 1.01])
    m = model([r])
    assert m['n'] == 8 and abs(m['median'] - 1.) < 1e-12
    assert abs(m['mad'] - .01) < 1e-12 and abs(m['margin'] - .03) < 1e-12
    assert gate(r, m, m['margin'])
    r['candidate'] = metric(.975)
    assert not gate(r, m, m['margin'])  # apparent 2.5% gain is below noise
    quiet = record([1.] * 8, .984)
    qm = model([quiet])
    assert qm['margin'] == .015 and gate(quiet, qm, qm['margin'])
    quiet['candidate'] = metric(.985)
    assert not gate(quiet, qm, qm['margin'])  # strictly below, not <=
    quiet['candidate'] = metric(.98, .99)
    assert not gate(quiet, qm, qm['margin'])
    # Capture-order disagreement is not averaged away.
    quiet['candidate'] = metric(.97)
    quiet['candidate']['orders']['BA']['paired'] = 1.01
    assert not gate(quiet, qm, qm['margin'])
    displaced = record([.97] * 8, .90)
    assert not model([displaced])['valid']
    biased = record([.992] * 8, .98)
    bm = model([biased])
    assert bm['valid'] and not gate(biased, bm, bm['margin'])
    quiet['post']['orders']['BA']['median'] = 1.03
    assert not model([quiet])['valid']
    too_few = record([1.] * 8)
    too_few['post']['panels'] = []
    try:
        model([too_few])
    except RuntimeError:
        pass
    else:
        raise AssertionError('fewer than six null captures accepted')
    bad = record([1.] * 7 + [float('nan')])
    try:
        model([bad])
    except RuntimeError:
        pass
    else:
        raise AssertionError('nonfinite null accepted')
    return dict(independent_null_n=8, raw_mad=.01, adaptive_margin=.03,
                strict_boundary_paired_order_drift_and_nonfinite_rejected=True)


def router_pack(tree):
    factory = function(tree, '_make_nb')
    scope = {'torch': torch}
    for name in ('_rt_pack', '_rt_arm_pack', '_rt_arm_call'):
        node = function(factory, name)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(CANDIDATE), 'exec'), scope)
    gen = torch.Generator().manual_seed(77673)
    w = torch.randn(256, 8192, generator=gen)
    hi = w.to(torch.bfloat16)
    lo = (w - hi.float()).to(torch.bfloat16)
    payloads = 0
    for s, wk in ((8, 8), (8, 16)):
        packed = scope['_rt_arm_pack'](w, s, wk)
        ku = 128 // s // wk
        assert packed.shape == (ku, s, 16, wk, 8, 32, 8) and packed.is_contiguous()
        # Invert the documented lane/tile permutation independently. Bitwise
        # expected hi/lo values follow the allowed FP32->BF16 residual split.
        units = packed.permute(2, 1, 3, 0, 4, 5, 6).contiguous().reshape(16, 128, 8, 32, 8)
        for half, reference in ((0, hi), (1, lo)):
            restored = (units[:, :, 4 * half:4 * (half + 1)]
                        .permute(0, 1, 3, 2, 4).reshape(16, 128, 8, 4, 4, 2, 2, 2)
                        .permute(0, 6, 2, 1, 3, 4, 5, 7).reshape(256, 8192))
            assert torch.equal(restored.view(torch.int16), reference.view(torch.int16))
            payloads += restored.numel()
    # The oracle is the public full-FP32 router equation, not either GPU path.
    x = torch.randn(16, 8192, generator=gen).to(torch.bfloat16).double()
    reference = x @ w.double().t()
    split = x @ hi.double().t() + x @ lo.double().t()
    relative = float((split - reference).norm() / reference.norm())
    assert relative < 1e-5  # stricter than TASK.md's 1e-4 router requirement
    calls = []
    scope['ext'] = lambda: SimpleNamespace(routertail_small=lambda *args: calls.append(args) or 0)
    scope['torch'] = SimpleNamespace(cuda=SimpleNamespace(current_stream=lambda: SimpleNamespace(cuda_stream=719)))
    x = torch.empty(16, 8192, dtype=torch.bfloat16)
    out = torch.empty(16, 256)
    scope['_rt_arm_call'](x, out, packed, 8, 16, 2)
    assert calls == [(packed.data_ptr(), x.data_ptr(), out.data_ptr(), 16, 8, 16, 719, 2)]
    return dict(bf16_payloads_roundtripped=payloads, geometries=2, fp64_router_relative=relative,
                launch_argument_transport=True)


def bounded_rq(tree, ns):
    factory = function(tree, '_make_chenwainuo')
    variant = ('n256s5m1hzm', 256, 5, 1, 0, 1, 0, 67)
    scope = dict(torch=torch, RQV=[variant], RQI={variant[0]: 0},
                 _DC_RQ_SPECS=ns['_DC_RQ_SPECS'], _rqmod=[None, None],
                 time=SimpleNamespace(monotonic=lambda: 0.))
    start = next(i for i, n in enumerate(factory.body) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == '_DTRQ_NAMES' for t in n.targets))
    stop = next(i for i, n in enumerate(factory.body) if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == '_DTGX' for t in n.targets))
    nodes = factory.body[start:stop]
    nodes += [n for n in factory.body if isinstance(n, ast.Assign)
              and any(isinstance(t, ast.Name) and t.id == 'RQFLAGS' for t in n.targets)]
    nodes += [function(factory, name) for name in ('_dtplans', '_dtstart', '_rqext')]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(CANDIDATE), 'exec'), scope)
    assert len(scope['_DTRQ']) == 4
    builds = []
    scope['_rqsrc'] = lambda variants: 'four exact templates'
    scope['load_inline'] = lambda *args, **kw: builds.append(kw) or SimpleNamespace(rqrun=True)
    jobs = scope['_dtstart']()
    assert set(jobs) == {'rq'}
    jobs['rq']['thread'].join()
    assert 'mod' in jobs['rq'] and len(builds) == 1
    assert '-std=c++20' in builds[0]['extra_cflags']
    assert '-std=c++20' in builds[0]['extra_cuda_cflags']
    assert '-gencode=arch=compute_90a,code=sm_90a' in builds[0]['extra_cuda_cflags']

    class TorchProxy:
        cuda = SimpleNamespace(get_device_properties=lambda d: SimpleNamespace(multi_processor_count=132))
        def __getattr__(self, name):
            return getattr(torch, name)

    scope['torch'] = TorchProxy()
    w = torch.arange(128 * 64).reshape(128, 64).to(torch.bfloat16)
    pk = (torch.zeros(1, 1, dtype=torch.uint8), torch.zeros(1), torch.zeros(1))
    scope['_rqpack'] = lambda weights: pk
    scope['_rqunpack'] = lambda *args: w.view(torch.int16).int() & 65535
    scope['_rqesc'] = scope['_rqesc2'] = lambda *args: (torch.zeros(1), 0)
    st = {'qkv_gate_raw': w, 'pooler_raw': w}
    actual = []
    for (layer, m), (name, ks) in ns['_DC_RQ_SPECS'].items():
        plans = list(scope['_dtplans'](st, layer, m, 128, jobs, float('inf')))
        assert len(plans) == 1 and plans[0][0] == name + '-k%d' % ks
        assert plans[0][1]['p']['KS'] == ks and not bool(plans[0][1]['p']['cnt'].any())
        actual.append(plans[0][0])
    assert not list(scope['_dtplans'](st, 'pooler', 16, 128, jobs, float('inf')))
    # The preexisting exact roundtrip guard must still veto bad packed weights.
    st.pop('dt_rqpk')
    scope['_rqunpack'] = lambda *args: torch.zeros_like(w, dtype=torch.int32)
    assert not list(scope['_dtplans'](st, 'qkv_gate', 32, 128, jobs, float('inf')))
    return dict(exact_board_tags=actual, rq_only_build=True, cxx20_sm90a_flags=True,
                off_scope_and_bad_pack_rejected=True)


def protocol(ns):
    validate = ns['_dc_validate']
    results = []

    def run(**kwargs):
        backend = legacy.install(ns, **kwargs)
        seeds = []
        ns['_dc_validate'] = lambda *a: seeds.append(a[-1])
        ns['_dc_parent_kernel_fn'] = backend.function()
        ns['_dc_call'] = lambda d, s, arm: backend.function(True)(d, s)
        report = ns['_dc_empty_report']()
        ok = ns['_dc_evaluate']('o', 1, {}, {'o': torch.empty(32, 64)}, {'tag': 'test'},
                                 None, None, float('inf'), report)
        assert ok == report['admitted']
        assert backend.begins == backend.ends == len(backend.graphs)
        assert backend.capturing is None and ns['_NG_TLS'].lease is None
        for i in range(len(backend.graphs) // 24):
            group = backend.graphs[24 * i:24 * (i + 1)]
            assert len({op[2].data_ptr() for g in group for op in g.ops}) == 1
            assert len({op[1].data_ptr() for g in group for op in g.ops}) == 1
            assert all(op[2].data_ptr() % 65536 == op[1].data_ptr() % 65536 == 0 for g in group for op in g.ops)
            assert all(len(g.ops) == 2 for g in group[:8] + group[16:])
            assert sorted(len(g.ops) for g in group[8:16]) == [1] * 4 + [2] * 4
        return report, backend, seeds

    r, b, seeds = run(candidate=9., slot_bias=.05, startup=40.)
    assert r['admitted'] and r['n'] == 16 and r['confirm'].endswith(':pass')
    assert len(seeds) == len(set(seeds)) == 3 and len(b.graphs) == 48
    results.append(dict(case='real gain, balanced slots and first-replay bias', **r))
    r, b, _ = run(candidate=10., address_penalty=.7, slot_bias=.05)
    assert not r['admitted'] and r['n'] == 8 and len(b.graphs) == 24 and r['confirm'] == 'skipped'
    results.append(dict(case='same computation rejects despite old address/ABBA artifacts', **r))
    r, b, _ = run(candidate=9.9)
    assert not r['admitted']
    results.append(dict(case='sub-1.5-percent gain rejected', **r))
    r, b, _ = run(candidate=9., capture_bias=.05)
    assert not r['admitted'] and r['null_mad'] > .04
    results.append(dict(case='opposing capture biases widen MAD, not canceled', **r))
    # A fresh confirmation that loses must retract even a strong screening win.
    b = legacy.install(ns, candidate=9.)
    ns['_dc_parent_kernel_fn'] = b.function()
    ns['_dc_call'] = lambda d, s, arm: b.function(True)(d, s)
    calls = []
    def confirm_loses(*args):
        calls.append(args[-1])
        if len(calls) == 2:
            b.candidate = 10.
    ns['_dc_validate'] = confirm_loses
    r = ns['_dc_empty_report']()
    assert not ns['_dc_evaluate']('o', 1, {}, {'o': torch.empty(32, 64)}, {}, None, None, float('inf'), r)
    assert r['confirm'].endswith(':fail') and len(b.graphs) == 48
    results.append(dict(case='fresh-seed confirmation veto', **r))
    # Borrowed output numerical checks are real checks, not alias comparisons.
    b = legacy.install(ns, candidate=8., wrong=True)
    ns['_dc_parent_kernel_fn'] = b.function()
    ns['_dc_call'] = lambda d, s, arm: b.function(True)(d, s)
    ns['_dc_validate'] = lambda *a: None
    try:
        ns['_dc_evaluate']('o', 1, {}, {'o': torch.empty(32, 64)}, {}, None, None, float('inf'), ns['_dc_empty_report']())
    except RuntimeError as error:
        assert 'accuracy' in str(error)
    else:
        raise AssertionError('wrong result admitted')
    assert not b.graphs and ns['_NG_TLS'].lease is None
    ns['_dc_validate'] = validate
    return results


def validator(ns, tree):
    # Independent analytical model: a diagonal/identity-like weight matrix makes
    # x @ W.T exact in BF16 on normal, outlier, one-hot and zero inputs.
    b = legacy.install(ns)
    ns['_dt_inputs'] = ns.pop('_real_inputs')
    ns['_V6_K'] = {'o': 64}
    ns['_v6_n'] = lambda l, m: 32
    w = torch.eye(32, 64, dtype=torch.bfloat16)
    ns['_dc_parent_kernel_fn'] = lambda d, s: (d['x'].double() @ w.double().t()).to(torch.bfloat16)
    real_call = ns['_dc_call']
    ns['_dc_call'] = lambda d, s, a: ns['_dc_parent_kernel_fn'](d, s)
    ns['_dc_validate']('o', 8, {}, {'o': w}, {'kind': 'rt'}, float('inf'), 717)
    ns['_dc_call'] = lambda d, s, a: ns['_dc_parent_kernel_fn'](d, s) + 16
    try:
        ns['_dc_validate']('o', 8, {}, {'o': w}, {'kind': 'rt'}, float('inf'), 717)
    except RuntimeError as error:
        assert 'accuracy' in str(error)
    else:
        raise AssertionError('FP64 validator accepts perturbed identity output')
    ns['_dc_call'] = real_call
    return dict(fp64_identity_oracle=True, perturbed_output_rejected=True)


def prepare_wiring(ns, tree):
    backend = legacy.install(ns)
    backend_torch = ns['_dt_torch']
    backend_torch.cuda.get_device_capability = lambda dev: (9, 0)
    backend_torch.ones = lambda *a, **kw: torch.ones(8)
    ns['_ctx_bind'] = lambda x: None
    ns['_K']['chenwainuo']['dtstart'] = lambda: {'rq': {}}
    ns['_dc_parent_prepare'] = lambda w: {'chenwainuo': {'dt_rqpk': 'scratch'}, 'routes': {'off-pin': 'kept'}}
    seen = []
    ns['_dc_plan'] = lambda l, m, *args: dict(tag='%s%d' % (l, m))
    def evaluate(l, m, s, w, arm, flush, sink, deadline, report):
        seen.append((l, m))
        report.update(reason='pass' if l == 'router' else 'no-gain', admitted=l == 'router')
        return report['admitted']
    ns['_dc_evaluate'] = evaluate
    weights = {'o': SimpleNamespace(is_cuda=True, device='cpu')}
    with contextlib.redirect_stdout(io.StringIO()) as out:
        state = ns['prepare'](weights)
    assert seen == list(ns['_DC_CELLS']) and len(seen) == 6
    assert set(state['dense_admit']) == set(ns['_DC_RT_SPECS'])
    assert state['routes'] == {'off-pin': 'kept'} and 'dt_rqpk' not in state['chenwainuo']
    assert out.getvalue().count('DENSE_ADMIT ') == 6
    assert ns['_NG_TLS'].lease is None
    # No static inheritance of historical board DT wins: all fail closed.
    ns['_dc_evaluate'] = lambda *args: False
    with contextlib.redirect_stdout(io.StringIO()):
        state = ns['prepare'](weights)
    assert not state['dense_admit']
    # A failure after one completed cell preserves only the fully confirmed arm.
    def fail(l, m, *args):
        if m == 16:
            raise ns['_DTBudget']()
        args[-1].update(admitted=True, reason='pass')
        return True
    ns['_dc_evaluate'] = fail
    with contextlib.redirect_stdout(io.StringIO()):
        state = ns['prepare'](weights)
    assert set(state['dense_admit']) == {('router', 8)} and ns['_NG_TLS'].lease is None
    # Production output allocation is fresh, dispatch honors exact geometry.
    ns['_dc_call'] = lambda d, s, arm: torch.empty((d['config']['m'], d['config']['n']))
    ns['_dc_parent_kernel_fn'] = lambda d, s: 'default'
    ns['_v6_n'] = lambda l, m: 32
    ns['_V6_K'] = {'router': 64}
    data = dict(x=torch.empty(8, 64, dtype=torch.bfloat16), config=dict(layer='router', m=8, n=32, k=64))
    a, b = ns['kernel_fn'](data, state), ns['kernel_fn'](data, state)
    assert a.data_ptr() != b.data_ptr()
    data['config']['n'] = 31
    assert ns['kernel_fn'](data, state) == 'default'
    return dict(six_exact_cells=True, rejected_board_cells_retracted=True,
                budget_partial_cell_not_adopted=True, production_fresh_and_geometry_guarded=True)


def main():
    torch.set_num_threads(4)
    tree, ns = setup()
    ns['_real_inputs'] = ns['_dt_inputs']
    report = dict(candidate_sha256=hashlib.sha256(CANDIDATE.read_bytes()).hexdigest(),
                  lineage=lineage(tree), binding=legacy.binding_gates(ns),
                  scheduling=legacy.schedule_gates(ns), policy=policy(ns), router_pack=router_pack(tree),
                  bounded_rq=bounded_rq(tree, ns), protocol=protocol(ns),
                  numerical_validator=validator(ns, tree), prepare_wiring=prepare_wiring(ns, tree),
                  gpu='NOT RUN: no CUDA/nvcc; H200 compile, timing/nulls and prepare duration unverified')
    (ROOT / 'results/dense-consolidate-offline.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
