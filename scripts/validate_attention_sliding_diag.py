#!/usr/bin/env python3
"""CPU-only tests: never import the submission, torch, or its CUDA builders."""
import ast
import contextlib
import ctypes
import difflib
import hashlib
import io
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'board_best/attention_kb1h2_scored_2.6536.py'
CANDIDATE = ROOT / 'candidates/attention_sliding_diag.py'
base, candidate = BASE.read_bytes(), CANDIDATE.read_bytes()
assert hashlib.sha256(base).hexdigest() == '6b65dec7c1aaeb4675815555ddbb0ee8a0de17ab93e56523c3d1d386899998ca'
assert candidate.startswith(base), 'incumbent must be an exact byte prefix'
assert all(line == line.rstrip() for line in candidate[len(base):].splitlines()), 'new trailing whitespace'
tree, base_tree = ast.parse(candidate), ast.parse(base)
for i, node in enumerate(base_tree.body):
    assert ast.dump(node) == ast.dump(tree.body[i])
diff = list(difflib.unified_diff(base.decode().splitlines(), candidate.decode().splitlines()))
assert not any(line.startswith('-') and not line.startswith('---') for line in diff)
print('PASS AST parse, base SHA, unchanged incumbent AST, byte prefix, addition-only diff, new-line whitespace')

# Execute ONLY appended function definitions and literal constants, no CUDA
# import, tuner, prepare invocation, or candidate top-level effects.
ns = {}
for node in tree.body[len(base_tree.body):]:
    if isinstance(node, ast.FunctionDef):
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(CANDIDATE), 'exec'), ns)
    elif isinstance(node, ast.Assign):
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(CANDIDATE), 'exec'), ns)
assert len(set(ns['_ATTN_PROF_CASES'])) == 12
assert ns['_ATTN_PROF_REPS'] >= 8
assert sum(bool(w) for _, _, w in ns['_ATTN_PROF_CASES']) == 6
assert ns['_ATTN_PROF_LIMIT_S'] < 300
clean, stage = ns['_ap_clean'], ns['_ap_stage']
assert clean('void (anonymous namespace)::attn_1h<8, 3>(Params)') == 'attn_1h<8,3>'
for name in ('attn_1h<1,5>', 'attn_kernel<8,true>', 'vfg_kernel<true,false>',
             'b1vp2::vp2_kernel', 'wst_ct14b<3>', 'gather_kp'):
    assert stage(name) == 'attention', name
assert stage('vfgm2_kernel') == stage('vfgm_kernel') == 'merge'
for name in ('qk_prep', 'norm_gate', 'cache_write', 'rms_norm'):
    assert stage(name) == 'qkprep_norm_cachewrite'
assert stage('unrecognized_kernel') == 'other'
print('PASS official cases, >=8 replays, budget, actual incumbent kernel-name classification')

kv = ns['_ap_kv_bytes']
assert kv([80000], 1, 2048) == (2048 * 256, 2047 * 256)
assert kv([80000] * 32, 8, 2048) == (32 * 2055 * 256, 32 * 2047 * 256)
assert kv([74000, 86000], 8, 0) == ((160000 + 16) * 256, 160000 * 256)
assert kv([0, 10], 8, 2048) == (26 * 256, 10 * 256)
assert math.isclose(kv([80000], 1, 2048)[0] / 4.8e12 * 1e6, 0.10922666666666667)
print('PASS KV union floor (q8 not multiplied by 8), full spread, short context, units')

events, expected = [], []
for i in range(8):
    t = i * 1000.0
    chunk = [(t + 120, 10, 'vfg_kernel<true,false>'), (t + 132, 3, 'vfgm2_kernel')]
    events.extend([(t, 100, 'flush_a'), (t + 105, 10, 'flush_b')] + chunk)
    expected.append(chunk)
split = ns['_ap_split']
chunks = split(events, ('flush_a', 'flush_b'), 8)
assert chunks == expected
m = ns['_ap_metrics'](chunks)
assert m['total'] == 13 and m['span'] == 15 and m['gap'] == 2
assert m['idle'] == 2 and m['overlap'] == 0
assert m['kernels']['vfgm2_kernel'] == (3, 1)
assert m['stages'] == {'attention': 10, 'merge': 3}
overlap = ns['_ap_metrics']([[(0, 10, 'vfg_kernel'), (8, 3, 'vfgm2_kernel')]] * 8)
assert overlap['total'] == 13 and overlap['span'] == 11 and overlap['gap'] == -2
assert overlap['idle'] == 0 and overlap['overlap'] == 2
for broken in (events[1:], events[:-2], events + [(9999, 1, 'flush_a')],
               [(0, 1, 'flush_a'), (1, 1, 'flush_b')] * 8):
    try:
        split(broken, ('flush_a', 'flush_b'), 8)
    except ValueError:
        pass
    else:
        raise AssertionError('bad or graph-missing trace accepted')
print('PASS multi-kernel flush exclusion, eight complete graphs, gap/overlap, missing-trace rejection')

results = {}
for B, L, w in ns['_ATTN_PROF_CASES']:
    results[B, L, w] = dict(span=15.0, gap=2.0, stages={'attention': 10.0, 'merge': 3.0}, window=w)
out = io.StringIO()
with contextlib.redirect_stdout(out):
    ns['_ap_summary'](results)
text = out.getvalue()
assert text.count('SUMMARY case=') == 12 and text.count('SUMMARY load=') == 6
assert 'complete=6/6 cases=12/12' in text
assert 'weighted_us=5580.000' in text  # 6 * (46+16) * 15
assert 'weighted_potential_us=1860.000' in text  # 6 * 62 * (2+3)
assert 'half_combined_board_gain_x=1.200000' in text  # 15/(15-2.5)
with contextlib.redirect_stdout(io.StringIO()) as partial:
    ns['_ap_summary']({(1, 1, 2048): results[1, 1, 2048]})
assert 'SUMMARY case=d1S' in partial.getvalue()
assert 'half_combined_board_gain_x=NA-incomplete' in partial.getvalue()
assert 'weighted_potential_us=230.000' in partial.getvalue()
print('PASS 46S/16F weighted savings, six-load geometric gain, incomplete summary not imputed')

# Exercise native handle widths, node enumeration/topology, sink selection and
# reversible enable calls against a ctypes-shaped fake (no native GPU calls).
class FakeFunction:
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, *args):
        return self.fn(*args)


class FakeRuntime:
    def __init__(self, edges):
        self.calls = []

        def nodes(graph, dst, n):
            n._obj.value = 2
            if dst is not None:
                dst[0], dst[1] = 0x100000001, 0x100000002
            return 0

        def get_edges(graph, src, dst, n):
            n._obj.value = len(edges)
            if src is not None:
                for i, (a, b) in enumerate(edges):
                    src[i], dst[i] = a, b
            return 0

        def kind(node, dst):
            dst._obj.value = 0
            return 0

        def enabled(*args):
            self.calls.append(args)
            return 0

        for name, fn in (('cudaGraphGetNodes', nodes), ('cudaGraphGetEdges', get_edges),
                         ('cudaGraphNodeGetType', kind), ('cudaGraphNodeSetEnabled', enabled)):
            setattr(self, name, FakeFunction(fn))


graph = SimpleNamespace(raw_cuda_graph=lambda: 0x200000001, raw_cuda_graph_exec=lambda: 0x300000001)
runtime = FakeRuntime([(0x100000001, 0x100000002)])
with patch.object(ctypes, 'CDLL', return_value=runtime):
    toggle = ns['_ap_merge_toggle'](graph, m)
    toggle(False)
    toggle(True)
assert runtime.calls == [(0x300000001, 0x100000002, 0), (0x300000001, 0x100000002, 1)]
with patch.object(ctypes, 'CDLL', return_value=FakeRuntime([])):
    try:
        ns['_ap_merge_toggle'](graph, m)
    except ValueError:
        pass
    else:
        raise AssertionError('ambiguous native graph accepted')
try:
    ns['_ap_merge_toggle'](graph, ns['_ap_metrics']([[(0, 10, 'attn_1h<8,3>')]]))
except ValueError:
    pass
else:
    raise AssertionError('fused kernel accepted for external merge removal')
print('PASS 64-bit CUDA handles, terminal merge node only, reversible disable, fused/ambiguous refusal')

# Smoke-test the real line formatter, including every case and the final
# summary, using scalar stand-ins. Leave room for INFO lines in the 16 KB tail.
def fake_data(B, L, w, device, deadline):
    return {'seq_lens': SimpleNamespace(tolist=lambda: [80000] * B)}


def fake_capture(data, deadline):
    return object(), object()


def unavailable(*args):
    raise ValueError('no_separate_merge: fused_or_unrecognized_graph')


ns.update(_at_data=fake_data, _ap_capture=fake_capture, _ap_merge_toggle=unavailable,
          _AT_SELECTED={}, _zh_route=lambda data: 'vfg')
transcript, formatted_results = io.StringIO(), {}
with contextlib.redirect_stdout(transcript):
    for B, L, w in ns['_ATTN_PROF_CASES']:
        chunks = [[(0, 10, 'attn_1h<8,3>')]]
        if w and L == 8 and B > 1:
            chunks = [[(0, 10, 'vfg_kernel<true,false>'), (12, 3, 'vfgm2_kernel')]]
        metric = ns['_ap_metrics'](chunks * 8)
        metric.update(event=metric['span'] + 4, control_event=metric['span'] + 3.9)
        ns['_ap_measure'] = lambda *args, metric=metric: metric.copy()
        formatted_results[B, L, w] = ns['_ap_case'](B, L, w, None, None, None, (), 1e9)
    ns['_ap_summary'](formatted_results)
log = transcript.getvalue()
assert len(log.encode()) < 14500, len(log.encode())
assert log.count('ATTN_PROF_TOTAL ') == 12 and log.count('ATTN_PROF_FLOOR ') == 12
assert log.count('ATTN_PROF_NOMERGE ') == 6
print('PASS all-case print contract; simulated transcript %d bytes plus INFO fits 16 KB tail' % len(log.encode()))

# Even a profiler failure followed by a stream-join failure must restore every
# incumbent cache object. This executes the actual try/finally, not its AST.
cache_names = ('_WS', '_ZH_WS', '_AH_WS', '_INV_FREQ', '_NUM_SMS')
saved_caches = {name: {'sentinel': object()} for name in cache_names}
ns.update(saved_caches)


def join_failure(stream):
    raise RuntimeError('synthetic stream join failure')


original = SimpleNamespace(wait_stream=join_failure)
cuda = SimpleNamespace(is_available=lambda: True, get_device_capability=lambda: (9, 0),
                       current_device=lambda: 0, get_device_properties=lambda device: SimpleNamespace(name='fake'),
                       current_stream=lambda device: original,
                       Stream=lambda **kw: SimpleNamespace(wait_stream=lambda stream: None),
                       stream=lambda stream: contextlib.nullcontext())
ns['torch'] = SimpleNamespace(cuda=cuda, device=lambda *args: 'cuda:0', __version__='fake',
                             float32=object(), ones=lambda *a, **kw: object(), empty=lambda *a, **kw: object(),
                             sum=lambda *a, **kw: None, no_grad=contextlib.nullcontext)
ns.update(_MOD=object(), _ZH_THREADS=[], _AT_BUILD_THREADS=[], _at_check=lambda deadline: None,
          _at_drain=lambda deadline: None, _ap_collect=unavailable)
with contextlib.redirect_stdout(io.StringIO()):
    try:
        ns['_ap_run'](1e9)
    except RuntimeError as exc:
        assert 'stream join failure' in str(exc)
    else:
        raise AssertionError('failure injection did not reach cleanup')
assert all(ns[name] is saved_caches[name] for name in cache_names)
print('PASS incumbent cache identity restoration on profiler + stream-join failure')
print('ALL OFFLINE CHECKS PASS; GPU/CUPTI/runtime compatibility and elapsed time NOT verified')
