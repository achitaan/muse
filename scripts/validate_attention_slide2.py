#!/usr/bin/env python3
"""CPU/source tests only: deliberately never import torch or the candidate."""
import ast
import contextlib
import io
import math
import statistics
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / 'candidates/attention_slide2.py'
BASE = ROOT / 'board_best/attention_kb1h2_scored_2.6536.py'
text, base_text = PATH.read_text(), BASE.read_text()
tree, base_tree = ast.parse(text), ast.parse(base_text)
assert PATH.stat().st_size < 1_000_000
assert not any(x in text for x in ('subprocess', 'multiprocessing', 'Popen', 'os.system',
                                   'cudaGraphInstantiate_v2', 'cudaGraphInstantiate_v3'))
ns, original = {}, {}
source_names = ('CUDA_SRC', 'CPP_SRC', 'AH_CUDA', 'AH_CPP', 'ZH_WST_CUDA', 'ZH_WST_CPP',
                'ZH_VFG_CUDA', 'ZH_VFG_CPP')
for nodes, target in ((tree.body, ns), (base_tree.body, original)):
    for node in nodes:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in source_names
                                               for t in node.targets):
            exec(compile(ast.Module(body=[node], type_ignores=[]), '<literal>', 'exec'), target)
assert all(ns[k] == original[k] for k in source_names)
# Production prefix AST differs ONLY in explicit build-language flags, not
# incumbent kernel sources, routing, tuning or validation.
allowed = {'_build', '_zh_build_wst', '_zh_build_vfg', '_ah_build'}
for old, new in zip(base_tree.body, tree.body):
    if isinstance(old, ast.FunctionDef) and old.name in allowed:
        continue
    assert ast.dump(old) == ast.dump(new), getattr(old, 'name', type(old).__name__)
print('PASS AST, size, forbidden APIs, unchanged incumbent CUDA and non-build prefix AST')

load_calls = []
for node in ast.walk(tree):
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ('load_inline', '_zh_load_inline'):
        kw = {k.arg: k.value for k in node.keywords}
        assert ast.literal_eval(kw['extra_cflags']) == ['-std=c++20']
        load_calls.append(node)
assert len(load_calls) == 6
assert text.count("'-std=c++20'") + text.count('"-std=c++20"') >= 12
print('PASS all six in-process load_inline calls explicitly use host C++20')

functions = {'_ph_replace', '_ph_pair_source', '_slide_source', '_slide_stats', '_slide_gate',
             '_slide_abba', '_slide_launch', '_slide_workspace', '_slide_key', '_ph_analyze',
             '_ph_print', '_slide_phase', '_slide_case', 'prepare', 'kernel_fn'}
for node in tree.body[len(base_tree.body):]:
    if isinstance(node, ast.Assign):
        try:
            value = ast.literal_eval(node.value)
        except (ValueError, TypeError):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                ns[target.id] = value
    elif isinstance(node, ast.FunctionDef) and node.name in functions:
        # Strip annotations for the final dispatcher, whose Inputs is torch-based.
        if node.name == 'kernel_fn':
            node.returns = None
            for arg in node.args.args:
                arg.annotation = None
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<selected-def>', 'exec'), ns)
    elif isinstance(node, ast.ClassDef) and node.name == '_SlideProxy':
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<proxy>', 'exec'), ns)
ns.update(_at_math=math, _at_stats=statistics)
src = ns['_slide_source']()
marked = ns['_ph_pair_source'](src)
assert src.count('int rope_positions;') == 1
assert src.count('const bool rope_cached') == 1
assert src.count('if (p.rope && !rope_cached)') == 1
assert src.index('const bool rope_cached') < src.index('  {\n  if constexpr (!WS) {', src.index('const bool rope_cached'))
assert src.count('csync<WS, NT>();') == ns['CUDA_SRC'].count('csync<WS, NT>();')
assert src.count('__syncthreads();') == ns['CUDA_SRC'].count('__syncthreads();')
assert 'seq >= 0 && seq < p.rope_positions' in src
assert '(inv_freq.numel() - 32) / 64' in src
assert 'packed.numel() - 32) % 64 == 0' in src
assert 'n <= 131072' in src
assert 'else if (tidL < 8) cp_async16(smem_u32(&S.ifreq' in src
assert 'if (tidL == 8) cp_async4(smem_u32(&S.scl[0])' in src
assert 'else if (tidL == 9) cp_async4(smem_u32(&S.scl[1])' in src
assert 'slide_make_rope<<<' in src and 'at::cuda::getCurrentCUDAStream()' in src
# Source-of-truth math is the incumbent's actual device expression; exact text
# equality ensures no changed constants, FMA sequence, angle rounding or trig.
start = ns['CUDA_SRC'].index('      const float ang = (float)(seq + j) * S.ifreq[i];')
end = ns['CUDA_SRC'].index('      S.cs[e] = cv;', start)
trig = ns['CUDA_SRC'][start:end]
assert trig in src
assert trig.replace('(float)(seq + j) * S.ifreq[i]', '(float)pos * packed[i]') in src
epilogue = ns['CUDA_SRC'][ns['CUDA_SRC'].index('  // ---- merge: rescale'):ns['CUDA_SRC'].index('\nvoid run(')]
assert epilogue in src, 'publication / merge / launch protocol changed'
assert marked.count('ph_stamp(p.dbg, 1)') == 1
assert marked.count('ph_stamp(p.dbg, 4)') == 2
assert marked.count('void phase_mark(torch::Tensor d)') == 1
print('PASS plain + marked CUDA generation, literal-identical trig, bounds fallback, unchanged barriers/publication')

# Layout expectations follow the original 32-dimension-pair RoPE ABI. Every
# position has two complete 128B rows, both aligned for 16B cp.async chunks.
for pos in (0, 1, 2047, 73999, 80000, 85999, 131071):
    cs = [32 + 64 * pos + 4 * lane for lane in range(8)]
    sn = [32 + 64 * pos + 32 + 4 * lane for lane in range(8)]
    assert all(4 * offset % 16 == 0 for offset in cs + sn)
    assert sorted(i for chunk in cs + sn for i in range(chunk, chunk + 4)) == list(range(32 + 64 * pos, 96 + 64 * pos))
    assert sn[-1] + 4 <= 32 + ns['_SLIDE_POSITIONS'] * 64
print('PASS table addressing, complete nonoverlapping coverage and copy alignment at boundary positions')

stats = lambda r, p=None, s=.002, lo=.998, hi=1.002: dict(ratio=r, paired=r if p is None else p,
                                                        spread=s, low=lo, high=hi)
gate = ns['_slide_gate']
assert gate(stats(1), stats(.98))
assert not gate(stats(1), stats(.989))
assert not gate(stats(1), stats(.98, p=.99))
assert not gate(stats(1.01), stats(.8))  # biased null: reject any apparent win
assert not gate(stats(1, lo=1.0001), stats(.8))
assert not gate(stats(1, s=.03), stats(.98))  # real win smaller than null spread
assert not gate(stats(1), stats(float('inf')))
assert not gate(stats(1), stats(.98, p=float('nan')))
assert not gate(stats(1, s=float('nan')), stats(.8))
out = ns['_slide_stats']([10] * 32, [9.8] * 32, [.98] * 16)
assert math.isclose(out['ratio'], .98) and out['paired'] == .98 and out['spread'] == 0
print('PASS null/spread gate, both ratio gates, noisy-null rejection and statistics')

calls = []
class Event:
    def record(self):
        calls.append('event')
    def elapsed_time(self, end):
        return .01

ns.update(torch=SimpleNamespace(cuda=SimpleNamespace(Event=lambda **k: Event()),
                                sum=lambda *a, **k: calls.append('read')),
          _at_check=lambda d: None, _at_wait=lambda e, d: None)
a = SimpleNamespace(replay=lambda: calls.append('a'))
b = SimpleNamespace(replay=lambda: calls.append('b'))
result = ns['_slide_abba'](a, b, None, None, 1e9)
assert calls.count('read') == 64
assert [x for x in calls if x in ('a', 'b')] == ['a', 'b', 'b', 'a'] * 16
for i, x in enumerate(calls):
    if x in ('a', 'b'):
        assert calls[i - 2:i] == ['read', 'event'] and calls[i + 1] == 'event'
assert result['ratio'] == result['paired'] == 1
print('PASS actual ABBA replay function: 16 blocks, every graph preceded by excluded READ flush')

recorded = []
mod = SimpleNamespace(run=lambda *args: recorded.append(args))
proxy = ns['_SlideProxy'](mod, 'packed', ('acc', 'sum', 'count'))
proxy.run(*range(28))
expected = list(range(28)); expected[8] = 'packed'; expected[10:13] = ('acc', 'sum', 'count')
assert recorded == [tuple(expected)]
key_fn = ns['_slide_key']
class Tensor:
    def __init__(self, shape, dtype, device='cuda:0', contiguous=True):
        self.shape, self.dtype, self.device, self.contiguous = shape, dtype, device, contiguous
    def is_contiguous(self):
        return self.contiguous
    def numel(self):
        return math.prod(self.shape)

cfg = dict(batch=1, q_len=1, window=2048, page_size=1, num_q_heads=16, num_kv_heads=2,
           head_dim=64, kv_scale=1.0, rope_theta=500000.0,
           qk_scale_factor=5.4730064863875, sm_scale=.125)
data = dict(config=cfg, qkv=Tensor((1, 1, 2304), 'bf16'),
            page_table=Tensor((1, 80001), 'i32'), seq_lens=Tensor((1,), 'i32'),
            cache_loc=Tensor((1,), 'i64'), k_cache=Tensor((80001, 1, 2, 64), 'e4'),
            v_cache=Tensor((80001, 1, 2, 64), 'e4'), k_scale=Tensor((), 'f32'), v_scale=Tensor((), 'f32'))
ns.update(_SLIDE_SELECTED={(1, 1, 2048): True}, _SLIDE_DEVICE='cuda:0',
          torch=SimpleNamespace(bfloat16='bf16', int32='i32', int64='i64', float32='f32', float8_e4m3fn='e4'))
assert key_fn(data) == (1, 1, 2048)
for field, wrong in (('page_size', 2), ('rope_theta', 10000), ('qk_scale_factor', 1),
                     ('num_kv_heads', 4), ('head_dim', 128), ('sm_scale', .25)):
    saved = cfg[field]; cfg[field] = wrong
    assert key_fn(data) is None, field
    cfg[field] = saved
for name in ('qkv', 'page_table', 'seq_lens', 'cache_loc', 'k_cache', 'v_cache'):
    data[name].contiguous = False
    assert key_fn(data) is None, name
    data[name].contiguous = True
for name in ('qkv', 'page_table', 'seq_lens', 'cache_loc', 'k_cache', 'v_cache', 'k_scale', 'v_scale'):
    data[name].device = 'cuda:1'
    assert key_fn(data) is None, name
    data[name].device = 'cuda:0'
data['seq_lens'].shape = (2,)
assert key_fn(data) is None
data['seq_lens'].shape = (1,)
data['k_scale'].shape = (2,)
assert key_fn(data) is None
print('PASS shape/config/device/dtype/stride guard and wrong-layout fallback')
ns.update(_SLIDE_SELECTED={}, _SLIDE_BASE=lambda d: 'incumbent', _slide_launch=lambda d: 'candidate')
assert ns['kernel_fn']({}) == 'incumbent'
ns.update(_SLIDE_SELECTED={(1, 1, 2048): True}, _slide_key=lambda d: None)
assert ns['kernel_fn']({}) == 'incumbent'
ns['_slide_key'] = lambda d: (1, 1, 2048)
assert ns['kernel_fn']({}) == 'candidate'
print('PASS proxy argument isolation, default incumbent and admitted-shape-only dispatch')

# Static control-flow contract: both passes precede return-all and phase timing
# cannot select a shape. Prepare does not publish winners until stream joining.
case = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_slide_case')
case_text = ast.unparse(case)
assert 'for pass_id in range(2)' in case_text and 'return all(passes)' in case_text
assert case_text.count('_slide_abba(') == 4
assert '_at_reset_cache(data, cache_bytes)' in case_text
assert '_at_score(data, output, pack, cache_bytes' in case_text
prepare = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'prepare'][-1]
prep_text = ast.unparse(prepare)
assert prep_text.index('original.wait_stream(stream)') < prep_text.index('_SLIDE_SELECTED = selected')
assert 'except Exception as exc:\n        _SLIDE_SELECTED = {}' in prep_text
print('PASS two-pass selection, judge/cache checks, publication after stream join, fail-closed prepare')
print('ALL CPU TESTS PASS; CUDA build, trig bit parity, correctness and speed require remote GPU check')
