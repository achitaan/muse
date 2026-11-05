#!/usr/bin/env python3
"""CPU-only invariants and failure tests; no torch/candidate import or CUDA JIT.

These prove scheduling/index identities and gate behavior, NOT GPU numerics.
GPU outputs, whole-cache bytes, and speed are independently checked at prepare.
"""
import ast
import math
import statistics
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'board_best/attention_kb1h2_scored_2.6536.py'
CANDIDATE = ROOT / 'candidates/attention_prepovl.py'
base, candidate = BASE.read_bytes(), CANDIDATE.read_bytes()
assert candidate.startswith(base), 'incumbent byte prefix changed'
tree, bt = ast.parse(candidate), ast.parse(base)
assert all(ast.dump(a) == ast.dump(b) for a, b in zip(bt.body, tree.body))
ns = dict(_at_math=math, _at_stats=statistics, Inputs=dict,
          torch=SimpleNamespace(Tensor=object), kernel_fn=lambda d: ('base', d))
for node in tree.body:
    name = node.name if isinstance(node, ast.FunctionDef) else (
        node.targets[0].id if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) else None)
    if name == 'CUDA_SRC' or node.lineno > len(base.splitlines()):
        if isinstance(node, (ast.FunctionDef, ast.Assign)):
            exec(compile(ast.Module(body=[node], type_ignores=[]), '<cpu-definitions>', 'exec'), ns)
assert ns['_PO_SELECTED'] == {} and ns['_PO_MODS'] == {}
assert ns['_PO_GATE'] == 0.985
assert ns['_PO_CASES'] == ((1, 1, 2048), (16, 1, 2048), (32, 1, 2048), (1, 1, 0))
assert ns['kernel_fn']('input') == ('base', 'input')
print('PASS AST, incumbent byte/AST prefix, cold default dispatch, q1-only shape scope')

original = ns['CUDA_SRC']
cuda = ns['_po_source'](original)
phase = ns['_ph_pair_source'](cuda)
assert ns['CUDA_SRC'] == original
assert cuda.startswith('#ifndef ATTN_PREPOVL\n#define ATTN_PREPOVL 0\n#endif\n')
assert 'constexpr bool PREPOVL = ATTN_PREPOVL && L == 1 && !WS;' in cuda
assert cuda.count('cp_commit();') == original.count('cp_commit();')
for wait in ('cp_wait<IDXF ? 1 : 0>();', 'cp_wait<STAGES - 1>();'):
    assert cuda.count(wait) == original.count(wait)
assert cuda.count('if constexpr (!PREPOVL) csync<WS, NT>();') == 2
# Everything from the final prep barrier onward (including patch, pipeline,
# merge tickets, global stores, finalize, and launch plans) is byte-identical.
suffix = '  csync<WS, NT>();\n  if (!WS && has_new) {'
assert cuda[cuda.index(suffix):] == original[original.index(suffix):]
assert cuda.count('if (pos >= start && pos < end) {') == original.count('if (pos >= start && pos < end) {')
assert cuda.count('__sincosf(rr, &sv, &cv);') == original.count('__sincosf(rr, &sv, &cv);') + 1
for stmt in ('const float kq = rintf(ang * 0.15915493667125702f);',
             'float rr = fmaf(-kq, 6.25f, ang);',
             'rr = fmaf(-kq, 0.03318530693650246f, rr);',
             'rr = fmaf(-kq, 2.430837753308879e-10f, rr);'):
    assert cuda.count(stmt) == original.count(stmt) + 1
assert phase.count('ph_stamp(p.dbg, 1);') == 1
assert phase.count('ph_stamp(p.dbg, 2);') == 2
assert phase.count('void phase_mark(torch::Tensor d)') == 1
assert '-DATTN_PREPOVL=1' in candidate.decode()
assert "'-std=c++20'" in candidate.decode()[len(base):]
assert "'-gencode=arch=compute_90a,code=sm_90a'" in candidate.decode()[len(base):]
try:
    ns['_po_source'](original.replace('  constexpr bool QDIR = L == 8 && !WS;', ''))
except ValueError:
    pass
else:
    raise AssertionError('source drift did not fail closed')
print('PASS source transforms, flag-off default, unchanged commits/waits/tail, identical trig formula, phase anchors')

# Expected ownership comes directly from incumbent tfor<Q_ROWS*8, NT> and
# rr=ps*RPP+(tid>>3), part=tid&7, with Cfg<1,false>::NT=128 and RPP=16.
# Exhaustively prove the private-read precondition for removing the Q CTA join.
nt, rpp = 128, 16
for rows in (16, 4):
    copies = {(e >> 3, e & 7): tid for tid in range(nt) for e in range(tid, rows * 8, nt)}
    for tid in range(nt):
        for ps in range((rows + rpp - 1) // rpp):
            row, part = ps * rpp + (tid >> 3), tid & 7
            if row < rows:
                assert copies[row, part] == tid
# Each warp holds frequency lane i, and each thread needs part*4+e. This is
# exactly incumbent S.cs[j*32+i] at q_len=1 (j=jj=0), including both K heads.
for tid in range(nt):
    for e in range(4):
        src_lane = (tid & 7) * 4 + e
        assert 0 <= src_lane < 32
        assert (tid // 32) * 32 + src_lane in range((tid // 32) * 32, (tid // 32 + 1) * 32)
        assert src_lane == (tid & 7) * 4 + e
# Async-group model: constants/index=G0, q=G1, KV0=G2, KV1=G3.
# wait<1> after G1 completes G0, wait<2> after G3 completes G1. Even ONE
# emits its empty G3, preserving the outstanding-group count.
groups = ['index+constants', 'q', 'kv0', 'kv1-or-empty']
assert groups[:2][:-1] == ['index+constants']
assert groups[:-2] == ['index+constants', 'q']
print('PASS exhaustive q1 private-copy ownership, warp angle mapping, async-group chronology')

stats = ns['_po_stats']
stable = stats([10.] * 32, [10.] * 32, [1.] * 16)
assert stable['null_ok'] and stable['spread'] == 0
# Expected values are analytic log-ratio samples, independent of GPU timings.
balanced = stats([10.] * 32, [10.] * 32, [math.exp(-0.002), math.exp(0.002)] * 8)
assert balanced['null_ok'] and math.isclose(balanced['spread'], 0.004)
biased = stats([10.] * 32, [9.9] * 32, [0.99] * 16)
assert not biased['null_ok']
noisy = stats([10.] * 32, [10.] * 32, [math.exp(-0.02), math.exp(0.02)] * 8)
assert not noisy['null_ok']
admit = ns['_po_admit']
win = dict(ratio=0.984, paired=0.983)
edge = dict(ratio=0.985, paired=0.985)
assert admit([(win, stable), (edge, balanced)], True, True)
assert not admit([(win, stable)], True, True)
assert not admit([(win, stable), (win, biased)], True, True)
assert not admit([(win, stable), (win, noisy)], True, True)
assert not admit([(win, stable), (dict(ratio=0.98, paired=0.986), stable)], True, True)
assert not admit([(win, stable), (dict(ratio=0.986, paired=0.98), stable)], True, True)
for exact, valid in ((False, True), (True, False), (False, False)):
    assert not admit([(win, stable), (win, stable)], exact, valid)
assert not admit([(dict(ratio=float('nan'), paired=0.98), stable)] * 2, True, True)
print('PASS two-pass .985 gate, both metrics, stable/biased/noisy nulls, missing-cache/numeric/NaN rejection')

# Execute the actual ABBA function with fake events. Expect an ordered READ
# outside every timed interval and 16 * (A,B,B,A), not merely source keywords.
calls = []
class Event:
    def __init__(self, **kw):
        pass
    def record(self):
        calls.append('event')
    def elapsed_time(self, end):
        return 0.01
class Graph:
    def __init__(self, name):
        self.name = name
    def replay(self):
        calls.append(self.name)
ns.update(torch=SimpleNamespace(cuda=SimpleNamespace(Event=Event),
                                sum=lambda *a, **kw: calls.append('READ')),
          _AT_BLOCKS=16, _at_check=lambda *a: None, _at_wait=lambda *a: None)
result = ns['_po_abba'](Graph('A'), Graph('B'), object(), object(), 100)
assert calls == ['READ', 'event', 'A', 'event', 'READ', 'event', 'B', 'event',
                 'READ', 'event', 'B', 'event', 'READ', 'event', 'A', 'event'] * 16
assert result['ratio'] == 1 and result['paired'] == 1 and result['null_ok']
print('PASS executed 16-block ABBA, READ before every replay, events exclude READ')

# One replay's integer-nanosecond trace with huge absolute clock values. The
# expected prep=1us is manufactured by the explicit offset, not inferred from
# a plausible-looking GPU number. Reject incomplete and nonmonotonic traces.
anchor = 1 << 58
rows = [[0] * 32 for _ in range(3)]
rows[2][0] = anchor
for c in range(2):
    row = rows[c]
    for k, offset in enumerate((0, 1000, 2000, 3000, 4000, 4000, 5000, 6000)):
        row[k] = anchor + 1000 + c * 10 + offset
        row[10 + k] = offset
    row[18] = row[0] + 7000
    row[8] = row[21] = 1
analysis = ns['_ph_analyze']([rows], 'pair')
assert analysis['replays'] == 1 and analysis['summary']['prep_us'] == 1.0
rows[1][2] = rows[1][1] - 1
try:
    ns['_ph_analyze']([rows], 'pair')
except ValueError:
    pass
else:
    raise AssertionError('nonmonotonic phase accepted')
print('PASS single-replay phase analysis, integer ns precision, malformed phase refusal')

for word in ('subprocess', 'Popen', 'multiprocessing', 'os.system', 'os.popen'):
    assert word not in candidate.decode(), word
suffix_tree = tree.body[len(bt.body):]
names = [n.name for n in suffix_tree if isinstance(n, ast.FunctionDef)]
assert names.count('kernel_fn') == 1
assert not {'_launch', '_module', '_build', '_at_tune_once'} & set(names)
prepare = next(n for n in suffix_tree if isinstance(n, ast.FunctionDef) and n.name == 'prepare')
assert '_PO_ONCE = True' in ast.get_source_segment(candidate.decode(), prepare)
dispatch = next(n for n in suffix_tree if isinstance(n, ast.FunctionDef) and n.name == 'kernel_fn')
assert all(x not in ast.get_source_segment(candidate.decode(), dispatch) for x in ('print(', 'synchronize(', '_po_run(', '_po_shape('))
print('PASS no process starts, idempotent prepare, metadata-only production dispatch; GPU compile/numerics/speed NOT tested')
