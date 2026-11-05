#!/usr/bin/env python3
"""Offline proof/tests; never import torch, the candidate, or a CUDA builder."""
import ast
import contextlib
import difflib
import hashlib
import io
import math
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'board_best/attention_kb1h2_scored_2.6536.py'
CANDIDATE = ROOT / 'candidates/attention_phase_diag.py'
base, candidate = BASE.read_bytes(), CANDIDATE.read_bytes()
assert hashlib.sha256(base).hexdigest() == '6b65dec7c1aaeb4675815555ddbb0ee8a0de17ab93e56523c3d1d386899998ca'
assert candidate.startswith(base), 'production must be an exact byte prefix'
bt, tree = ast.parse(base), ast.parse(candidate)
assert all(ast.dump(a) == ast.dump(b) for a, b in zip(bt.body, tree.body))
assert len(tree.body) > len(bt.body)
assert not any(line.startswith('-') and not line.startswith('---') for line in
               difflib.unified_diff(base.decode().splitlines(), candidate.decode().splitlines()))
assert all(line == line.rstrip() for line in candidate[len(base):].splitlines())
ns = {}
for node in bt.body:
    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in
            ('CUDA_SRC', 'CPP_SRC', 'AH_CUDA', 'AH_CPP', 'ZH_VFG_CUDA', 'ZH_VFG_CPP') for t in node.targets):
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<base-literal>', 'exec'), ns)
for node in tree.body[len(bt.body):]:
    if isinstance(node, (ast.FunctionDef, ast.Assign)):
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<diagnostic-def>', 'exec'), ns)
assert ns['_ATTN_PHASE_REPS'] >= 8
assert len(ns['_ATTN_PHASE_CASES']) == 8
assert sum(bool(w) for _, _, w in ns['_ATTN_PHASE_CASES']) == 6
assert 85 < ns['_ATTN_PHASE_LIMIT_S'] <= 280
assert not any(isinstance(n, ast.FunctionDef) and n.name in
               ('kernel_fn', '_launch', '_ah_launch', '_module', '_build') for n in tree.body[len(bt.body):])
print('PASS AST, SHA256, exact incumbent byte-prefix/AST, addition-only diff, no production redefinitions')

source_copies = {k: ns[k] for k in ('CUDA_SRC', 'AH_CUDA', 'ZH_VFG_CUDA')}
specs = ns['_ph_build_specs']()
assert {k: ns[k] for k in source_copies} == source_copies
assert set(specs) == {'pair', '1h', 'vp2'}
for kind, (cpp, cuda, functions, flags) in specs.items():
    assert flags.count('-std=c++20') == 1
    assert '-gencode=arch=compute_90a,code=sm_90a' in flags
    assert '--use_fast_math' not in flags
    assert 'reinterpret_cast<long long*>(d.data_ptr<int64_t>())' in cuda
    assert cuda.count('__global__ void ph_anchor_kernel') == 1
    assert cuda.count('void phase_mark(torch::Tensor d)') == 1
    assert '%%globaltimer' in cuda and 'clock64()' in cuda
    assert 'row[10 + k]' in cuda and 'row[20] = sm' in cuda
    if kind != 'vp2':
        assert cuda.count('ph_stamp(p.dbg, 0)') == 1
        assert cuda.count('ph_stamp(p.dbg, 1)') == 1
        assert cuda.count('ph_stamp(p.dbg, 2)') == 2  # separate q1/q8 compile-time paths
        assert cuda.count('ph_stamp(p.dbg, 3)') == 1
        assert cuda.count('ph_stamp(p.dbg, 4)') == 2  # cooperative/plain branches
        assert cuda.count('ph_stamp(p.dbg, 5)') == 2
        assert cuda.count('ph_stamp(p.dbg, 6)') == 2
        assert cuda.count('ph_stamp(p.dbg, 7)') == 1
        assert cuda.count('ph_stamp(p.dbg, 18)') == 2  # active + empty return
        assert '* 16 + 6] = 1;' not in cuda
        assert 'ph_flag(p.dbg, 21, r_lo < r_hi)' in cuda
        assert 'p.dbg[(long long)ph_cta()*32+21]' in cuda
        ticket = cuda.index('ph_stamp(p.dbg, 4)')
        assert 'atom.acq_rel.gpu.global.add.s32' in cuda[ticket-180:ticket]
    else:
        assert 'vfgns::ph_anchor_kernel<<<' in cuda
        assert 'ph_stamp(pd, 3, 128)' in cuda and 'ph_stamp(pd, 4, 128)' in cuda
        assert 'bar.sync 5, 128' in cuda
        assert '#include <ATen/cuda/CUDAContext.h>' in cuda
        assert 'pd[(long long)ph_cta()*32+2] = pd[(long long)ph_cta()*32+1]' in cuda
try:
    ns['_ph_pair_source'](source_copies['CUDA_SRC'].replace('  const int split = first ? 0 : 1;', ''))
except ValueError as exc:
    assert 'anchor count' in str(exc)
else:
    raise AssertionError('changed incumbent source silently accepted')
print('PASS all three instrumented CUDA source generators, phase coverage, 64-bit pointers, C++20/SM90a flags, source drift refusal')

# Follow the actual selected incumbent route, not assumed winners from an
# earlier check. Diagnostic trace rows must match the host launch grid exactly.
cfg = dict(batch=32, q_len=8, num_kv_heads=2, window=2048, page_size=1)
data = dict(config=cfg, page_table=SimpleNamespace(shape=(32, 86008)), qkv=SimpleNamespace(device='cuda:0'))
key = (32, 8, 2048)
ns.update(_AT_SELECTED={key: ('1h-s4', '1h', dict(n_splits=4, coop=1))}, _at_key=lambda d: key,
          _zh_route=lambda d: None, _TUNED={},
          _plan=lambda *a: dict(n_splits=8, flat=0, cs=0, ncl=1, nc=1))
assert ns['_ph_route'](data) == ('1h', dict(n_splits=4, coop=1), 256)
ns['_AT_SELECTED'] = {}
assert ns['_ph_route'](data)[::2] == ('pair', 256)
cfg['batch'] = 1
ns['_zh_route'] = lambda d: 'vp2'
assert ns['_ph_route'](data) == ('vp2', {}, 32)
ns['_zh_route'] = lambda d: 'wst'
try:
    ns['_ph_route'](data)
except ValueError as exc:
    assert 'unimplemented' in str(exc)
else:
    raise AssertionError('unknown route silently profiled as paired-head')
print('PASS selected one-head/base/VP2 routing, exact trace grid sizes, unsupported-route refusal')


def make_records(vp2=False, coop=False):
    # Huge absolute nanoseconds deliberately exceed exact float64 integer
    # precision. Expected sub-us differences must survive integer subtraction.
    all_rows = []
    n = 32
    for replay in range(8):
        anchor = (1 << 58) + replay * 100000
        rows = [[0] * 32 for _ in range(n + 1)]
        rows[n][0] = anchor
        leader = 1 + replay % 4
        for c in range(n):
            row = rows[c]
            row[0] = anchor + 1000 + 10 * c
            row[18] = row[0] + 10000
            row[8] = int(c == leader and not vp2)
            row[9] = 1
            row[19] = int(coop)
            # True full sample set: request0/head0 plus middle and last.
            if c < 8 or c in (16, 31):
                offsets = (0, 1000, 1000 if vp2 else 2000, 5000, 6000,
                           5500 if vp2 else 6500, 8000, 9000)
                for k, offset in enumerate(offsets):
                    row[k] = row[0] + offset
                    row[10 + k] = 100000 + offset * 2
                if not vp2 and not coop and c != leader:
                    row[6] = row[7] = row[16] = row[17] = 0
        all_rows.append(rows)
    return all_rows


analyze = ns['_ph_analyze']
for kind, coop in (('pair', False), ('1h', True), ('vp2', False)):
    records = make_records(kind == 'vp2', coop)
    result = analyze(records, kind)
    s = result['summary']
    assert result['replays'] == 8 and result['ncta'] == 32
    assert s['launch_marker_to_first_us'] == 1.0
    assert s['start_skew_us'] == 0.31
    assert s['grid_span_us'] == 10.31
    assert math.isclose(s['start_skew_pct'], 100 * 0.31 / 10.31)
    assert s['prep_us'] == 1.0 and s['owner_residence_us'] == 10.0
    assert s['publication_us'] == 1.0 and s['gate_us'] == 1.0
    assert s['wait_setup_us'] == (-0.5 if kind == 'vp2' else 0.5)
    assert s['first_kv_wait_us'] == (0.0 if kind == 'vp2' else 1.0)
    assert s['merge_wait_us'] == (2.5 if kind == 'vp2' else 1.5)
    assert all(count == 8 for count in result['counts'].values())
    assert ('leader' in result['samples']) == (kind != 'vp2')
    if kind != 'vp2':
        assert result['cta_ids']['leader'] == {1, 2, 3, 4}
    assert result['samples']['first']['entry_us'] == 0
    if kind == 'vp2' or coop:
        assert result['samples']['first']['gate_kcycles'] == 18

broken = make_records()
broken[0][1][18] = 0
try:
    analyze(broken, 'pair')
except ValueError as exc:
    assert 'incomplete CTA' in str(exc)
else:
    raise AssertionError('missing CTA exit accepted')
broken = make_records()
broken[0][1][3] = broken[0][1][2] - 1
try:
    analyze(broken, 'pair')
except ValueError as exc:
    assert 'nonmonotonic' in str(exc)
else:
    raise AssertionError('reordered phase accepted')
try:
    analyze(make_records()[:7], 'pair')
except ValueError as exc:
    assert 'eight' in str(exc)
else:
    raise AssertionError('incomplete replay set accepted')
print('PASS 8 replay medians, dynamic-leader aggregation, all-CTA skew, integer ns precision, VP2 signed overlap, incomplete/reordered rejection')

transcript = io.StringIO()
with contextlib.redirect_stdout(transcript):
    for B, L, w in ns['_ATTN_PHASE_CASES']:
        kind = 'vp2' if (B, L, w) == (1, 8, 2048) else 'pair'
        label = ('d' if L == 1 else 'v') + str(B) + ('S' if w else 'F')
        ns['_ph_print'](label, kind, {}, analyze(make_records(kind == 'vp2'), kind),
                        dict(raw=[10] * 8, control=[12] * 8, instrumented=[13] * 8), 'fake')
text = transcript.getvalue()
assert text.count('ATTN_PHASE_SUMMARY ') == 8 and text.count('ATTN_PHASE_OVERHEAD ') == 8
assert text.count('role=leader n=8') == 7
assert 'instrumentation_us=1.000 instrumentation_pct=8.33 marker_envelope_us=2.000' in text
assert len(text.encode()) < 15500, len(text.encode())
print('PASS output contract, signed VP2 gap, paired overhead formula; all-case transcript %d bytes' % len(text.encode()))

# Simulate the actual replay loop: every graph, including the incumbent-only
# control, must have a fresh 512 MiB READ before its timing interval.
calls = []


class Event:
    def record(self):
        calls.append('event')

    def elapsed_time(self, end):
        return 0.01


trace = SimpleNamespace(zero_=lambda: calls.append('clear'),
                        cpu=lambda: SimpleNamespace(tolist=lambda: make_records()[0]))
graphs = [SimpleNamespace(replay=lambda name=name: calls.append(name)) for name in ('raw', 'control', 'instrumented')]
ns.update(torch=SimpleNamespace(cuda=SimpleNamespace(Event=lambda **kw: Event()),
                               sum=lambda *a, **kw: calls.append('read_flush')),
          _at_check=lambda deadline: None, _at_wait=lambda event, deadline: None)
times, records = ns['_ph_measure'](*graphs, trace, object(), object(), 1e9)
assert {key: len(value) for key, value in times.items()} == dict(raw=8, control=8, instrumented=8)
assert len(records) == 8 and calls.count('read_flush') == 24 and calls.count('clear') == 8
for i, call in enumerate(calls):
    if call in ('raw', 'control', 'instrumented'):
        assert calls[i-2:i] == ['read_flush', 'event'] and calls[i+1] == 'event'
print('PASS actual 8x ABBA/replay loop: cold READ before ALL 24 timed graphs; trace clears excluded from timing')

# Failure-inject the production-cache isolation finally block, including an
# error from the stream join itself; object identity must still be restored.
cache_names = ('_WS', '_ZH_WS', '_AH_WS', '_INV_FREQ', '_NUM_SMS')
saved = {name: {'sentinel': object()} for name in cache_names}
ns.update(saved)


def fail_join(stream):
    raise RuntimeError('fake stream join failure')


cuda = SimpleNamespace(get_device_capability=lambda: (9, 0), current_device=lambda: 0,
                       current_stream=lambda device: SimpleNamespace(wait_stream=fail_join),
                       Stream=lambda **kw: SimpleNamespace(wait_stream=lambda stream: None),
                       stream=lambda stream: contextlib.nullcontext())
ns.update(torch=SimpleNamespace(cuda=cuda, device=lambda *a: 'cuda:0', no_grad=contextlib.nullcontext,
                               float32=object(), ones=lambda *a, **kw: object(), empty=lambda *a, **kw: object()),
          _MOD=object(), _ZH_THREADS=[], _AT_BUILD_THREADS=[], _ph_builds=lambda deadline: {'pair': object()},
          _ph_case=lambda *a: (_ for _ in ()).throw(RuntimeError('fake phase failure')),
          _at_drain=lambda deadline: None)
with contextlib.redirect_stdout(io.StringIO()):
    try:
        ns['_ph_run'](1e9)
    except RuntimeError as exc:
        assert 'stream join failure' in str(exc)
    else:
        raise AssertionError('cleanup failure injection missed')
assert all(ns[name] is saved[name] for name in cache_names)
print('PASS production cache identity restoration on phase failure AND stream-join failure')
print('ALL CPU TESTS PASS; CUDA compilation, GPU correctness/overhead and runtime budget NOT locally verified')
