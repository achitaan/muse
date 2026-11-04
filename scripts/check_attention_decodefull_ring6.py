"""CPU-only checks for the bounded decode-full ring6 experiment.

No torch/CUDA installation is required. This verifies source identity, native
shared-memory layout, a randomized ring model, attention invariants, dispatch,
and admission policy. It is NOT a substitute for H200 compilation/conformance.
"""
import ast
from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import subprocess
import tempfile
from types import SimpleNamespace as NS


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / 'candidates/attention_decodefull_ring6.py'
INCUMBENT = ROOT / 'board_best/attention_kb1h2_scored_2.6536.py'
RESULTS = ROOT / 'results/attention_decodefull_ring6_cpu.json'


def assignments(tree):
    return {n.targets[0].id: n.value for n in tree.body
            if isinstance(n, ast.Assign) and len(n.targets) == 1
            and isinstance(n.targets[0], ast.Name)}


def functions(tree):
    return {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}


def load_functions(tree, names, ns):
    nodes = functions(tree)
    for name in names:
        exec(compile(ast.Module(body=[nodes[name]], type_ignores=[]), str(CANDIDATE), 'exec'), ns)


def source_checks(text, tree):
    old_tree = ast.parse(INCUMBENT.read_text())
    old = ast.literal_eval(assignments(old_tree)['CUDA_SRC'])
    new = ast.literal_eval(assignments(tree)['CUDA_SRC'])
    # Source of truth: the scored incumbent CUDA string, not a reconstructed copy.
    assert new == old
    ns = {'CUDA_SRC': new}
    load_functions(tree, ['_df_cuda_source'], ns)
    deep = ns['_df_cuda_source']()
    split = 'template <int L>\nvoid launch('
    # Every byte of device code (partition, math, barriers, merge, KV writes)
    # is identical except the single depth constant. Only the host launcher differs.
    expected = old[:old.index(split)].replace(
        'STAGES = F ? ((L == 1) ? 3 : 4) : 3;',
        'STAGES = F ? ((L == 1) ? 6 : 4) : 3;')
    assert deep[:deep.index('void launch_deep(')].rstrip() == expected.rstrip()
    assert 'else launch<8>' not in deep
    assert 'attn_kernel<1, true><<<' in deep
    assert 'resident >= 2' in deep and '2 * sm <= 227328' in deep
    assert deep.count('STAGES = F ? ((L == 1) ? 6 : 4) : 3;') == 1
    # The actual timed entrypoint has no timing, reference or host-sync operation.
    entry = ast.unparse(functions(tree)['kernel_fn'])
    for forbidden in ('synchronize', '.item(', 'sleep(', '_df_tune(', 'print(', '_at_data('):
        assert forbidden not in entry
    assert text.startswith('# okc: problem=muse_attention_decode\n')
    assert '\r' not in text
    return old, deep


def native_layout(source):
    start = source.index('template <int L, bool F>\nstruct Cfg')
    constants = source[start:source.index('struct Params {', start)]
    start = source.index('template <int L, bool F>\n    struct Smem')
    smem = source[start:source.index('\n};', start) + 3]
    preamble = '''#include <cstdint>
#include <cstdio>
#include <cstddef>
#define __device__
#define __forceinline__ inline
using __nv_bfloat16 = uint16_t;
'''
    code = preamble + constants + smem + '''
int main() {
  using S = Smem<1, true>;
  static_assert(offsetof(S, k) % 1024 == 0);
  static_assert(offsetof(S, v) % 1024 == 0);
  static_assert(2 * (sizeof(S) + 1024) <= 227328);
  printf("%zu %zu %zu %zu %zu\\n", sizeof(S), sizeof(Smem<1, false>),
         sizeof(Smem<8, true>), offsetof(S, v), offsetof(S, kvq));
}
'''
    cc = shutil.which('clang++') or shutil.which('g++')
    if cc is None:
        return {'status': 'skipped: no native C++ compiler'}
    # All generated files remain inside this lane's only writable directory.
    with tempfile.TemporaryDirectory(prefix='.ring6-layout-', dir=ROOT / 'results') as td:
        cpp, exe = Path(td) / 'layout.cpp', Path(td) / 'layout'
        cpp.write_text(code)
        subprocess.run([cc, '-std=c++20', '-O2', str(cpp), '-o', str(exe)], check=True)
        values = list(map(int, subprocess.check_output([str(exe)], text=True).split()))
    size, nonflat, verify, v_offset, q_offset = values
    return dict(status='passed', sizeof=size, dynamic_bytes=size + 1024,
                nonflat_sizeof=nonflat, verify_sizeof=verify,
                v_offset=v_offset, q_offset=q_offset,
                shared_residency=227328 // (size + 1024))


def ring_tiles(rows, new_row, seq, start, end, depth, seed):
    """Random completions of 128 producer arrivals; ordered consumer acquisition.

    Mirrors segment ROLE=1 cp_async_mbar_arrive and ROLE=2 ws_acquire/release.
    It deliberately starts at non-tile-aligned logical positions and patches new
    KV on the consumer, as the kernel does. Tuple rows stand for 128-byte KV rows.
    """
    rng = random.Random(seed)
    n = (end - start + 63) // 64
    slots = [None] * depth
    written = consumed = 0
    output = []
    while consumed < n:
        actions = []
        if written < n and slots[written % depth] is None:
            actions.append(('issue', written % depth))
        actions += [('arrive', i) for i, s in enumerate(slots) if s and s['left']]
        slot = slots[consumed % depth]
        if slot and slot['tile'] == consumed and slot['left'] == 0:
            actions.append(('consume', consumed % depth))
        assert actions, 'ring deadlock'
        action, st = rng.choice(actions)
        if action == 'issue':
            data = []
            for pos in range(start + written * 64, start + (written + 1) * 64):
                data.append(rows[pos] if pos < min(seq, end) else None if pos < end else (0, 0))
            slots[st] = dict(tile=written, phase=(written // depth) & 1, left=128, data=data)
            written += 1
        elif action == 'arrive':
            # Batch an arbitrary subset of producer thread arrivals.
            slots[st]['left'] -= rng.randrange(1, slots[st]['left'] + 1)
        else:
            assert slot['phase'] == ((consumed // depth) & 1)
            for i, row in enumerate(slot['data']):
                pos = start + consumed * 64 + i
                if seq <= pos < end:
                    row = new_row
                if pos < end:
                    output.append(row)
            # Every consumer thread arrives before the next producer overwrite.
            slots[st] = None
            consumed += 1
    return output


def fp8(x):
    """E4M3 finite value from one byte (PTX E4M3; 0x7f is NaN)."""
    sign, exponent, mantissa = -1 if x & 128 else 1, (x >> 3) & 15, x & 7
    assert (x & 127) != 127
    return sign * (mantissa * 2 ** -9 if exponent == 0 else (1 + mantissa / 8) * 2 ** (exponent - 7))


def numerical_model():
    rng = random.Random(20261104)
    schedules = 0
    for seq in (0, 1, 62, 63, 64, 65, 127, 128, 129, 257, 1023, 4097):
        physical = [(rng.getrandbits(64), rng.getrandbits(64)) for _ in range(seq)]
        rng.shuffle(physical)
        new = (rng.getrandbits(64), rng.getrandbits(64))
        for start in sorted({0, seq // 3, max(0, seq - 67)}):
            for end in sorted({max(start + 1, seq), seq + 1}):
                expected = physical[start:min(seq, end)] + ([new] if end > seq else [])
                for repeat in range(25):
                    a = ring_tiles(physical, new, seq, start, end, 3, repeat)
                    b = ring_tiles(physical, new, seq, start, end, 6, repeat + 100)
                    # Expected values come directly from logical-order semantics,
                    # not from a captured output of either ring implementation.
                    assert a == b == expected
                    schedules += 2
    # Independent dense softmax vs online tile recurrence, including all 16 GQA
    # query heads and FP8 KV. Source: the problem's softmax(QK/8)V equation.
    maximum_error = 0.0
    for seq in (1, 63, 64, 65, 129, 257):
        q = [[rng.gauss(0, 1) for _ in range(64)] for _ in range(16)]
        for row in q:
            norm = math.sqrt(sum(x * x for x in row) / 64 + 1e-5)
            row[:] = [x * 5.4730064863875 / norm for x in row]
        keys = [[[fp8(rng.randrange(0x20, 0x50)) for _ in range(64)] for _ in range(2)] for _ in range(seq + 1)]
        vals = [[[fp8(rng.randrange(0x20, 0x50)) * rng.choice((-1, 1)) for _ in range(64)] for _ in range(2)] for _ in range(seq + 1)]
        rows = list(zip(keys[:seq], vals[:seq]))
        new = (keys[-1], vals[-1])
        ordered = ring_tiles(rows, new, seq, 0, seq + 1, 6, seq)
        for h, query in enumerate(q):
            hk = h // 8
            logits = [sum(x * y for x, y in zip(query, k[hk])) / 8 for k in keys]
            exps = [math.exp(x - max(logits)) for x in logits]
            dense = [sum(w * v[hk][d] for w, v in zip(exps, vals)) / sum(exps) for d in range(64)]
            m, total, acc = -math.inf, 0.0, [0.0] * 64
            for t in range(0, len(ordered), 64):
                tile = ordered[t:t + 64]
                scores = [sum(x * y for x, y in zip(query, k[hk])) / 8 for k, _ in tile]
                nm = max(m, max(scores))
                alpha = math.exp(m - nm)
                weights = [math.exp(s - nm) for s in scores]
                total = total * alpha + sum(weights)
                acc = [a * alpha + sum(w * v[hk][d] for w, (_, v) in zip(weights, tile)) for d, a in enumerate(acc)]
                m = nm
            online = [a / total for a in acc]
            gate = [rng.gauss(0, 1) for _ in range(64)]
            def finish(a):
                norm = math.sqrt(sum(x * x for x in a) / 64 + 1e-5)
                return [x / norm / (1 + math.exp(-g)) for x, g in zip(a, gate)]
            err = max(abs(x - y) for x, y in zip(finish(dense), finish(online)))
            maximum_error = max(maximum_error, err)
            assert err < 1e-11
    return dict(randomized_ring_schedules=schedules, attention_max_abs_error=maximum_error)


class FakeTensor:
    def __init__(self, shape, dtype, device='cuda:0', pointer=256):
        self.shape, self.dtype, self.device = shape, dtype, device
        self.pointer, self.contiguous, self.is_cuda = pointer, True, True
        self.ndim = len(shape)

    def is_contiguous(self):
        return self.contiguous

    def numel(self):
        return math.prod(self.shape)

    def data_ptr(self):
        return self.pointer


def fake_data(B=16):
    cfg = dict(batch=B, q_len=1, window=0, page_size=1, num_q_heads=16,
               num_kv_heads=2, head_dim=64, rope_theta=0.0, kv_scale=1.0,
               qk_scale_factor=5.4730064863875, sm_scale=0.125)
    data = {'config': cfg}
    for name, shape, dtype in (
            ('qkv', (B, 1, 2304), 'bfloat16'), ('seq_lens', (B,), 'int32'),
            ('cache_loc', (B,), 'int64'), ('page_table', (B, 86001), 'int32'),
            ('k_cache', (B * 86001, 1, 2, 64), 'float8_e4m3fn'),
            ('v_cache', (B * 86001, 1, 2, 64), 'float8_e4m3fn'),
            ('k_scale', (), 'float32'), ('v_scale', (), 'float32')):
        data[name] = FakeTensor(shape, dtype)
    return data


def policy_checks(tree):
    torch = NS(**{k: k for k in ('bfloat16', 'int32', 'int64', 'float8_e4m3fn', 'float32')})
    torch.Tensor = FakeTensor
    ns = {'torch': torch, 'Inputs': dict}
    load_functions(tree, ['_df_layout', 'kernel_fn'], ns)
    for B in (16, 32):
        assert ns['_df_layout'](fake_data(B))
    rejected = 0
    for name, value in (('batch', 1), ('q_len', 8), ('window', 2048), ('page_size', 64),
                        ('num_q_heads', 32), ('num_kv_heads', 4), ('head_dim', 128),
                        ('rope_theta', 500000.0), ('kv_scale', 1 / 64)):
        data = fake_data()
        data['config'][name] = value
        assert not ns['_df_layout'](data)
        rejected += 1
    for name in ('qkv', 'seq_lens', 'cache_loc', 'page_table', 'k_cache', 'v_cache', 'k_scale', 'v_scale'):
        for field, value in (('device', 'cuda:1'), ('pointer', 1), ('dtype', 'wrong'), ('contiguous', False)):
            data = fake_data()
            setattr(data[name], field, value)
            # qkv device is the comparison reference; make other tensors differ.
            assert not ns['_df_layout'](data)
            rejected += 1
    ns.update(_DF_SELECTED={16}, _DF_DEVICE='cuda:0',
              _df_launch=lambda data: 'ring6', _DF_INCUMBENT=lambda data: 'incumbent')
    assert ns['kernel_fn'](fake_data()) == 'ring6'
    assert ns['kernel_fn'](fake_data(32)) == 'incumbent'
    data = fake_data()
    data['config']['q_len'] = 8
    assert ns['kernel_fn'](data) == 'incumbent'
    ns.update(_at_math=math, _DF_GATE=0.988)
    load_functions(tree, ['_df_accept_timing', '_df_equiv'], ns)
    # Source: the candidate's documented 1.2% gross graph gate on BOTH metrics.
    for ratio, paired, expected in ((0.98, 0.98, True), (0.988, 0.988, True),
                                     (0.989, 0.98, False), (0.98, 0.989, False),
                                     (math.nan, 0.98, False), (0.98, math.inf, False),
                                     (-1.0, 0.98, False), (0.0, 0.98, False)):
        assert ns['_df_accept_timing'](ratio, paired) == expected
    for judge_ok, cache_ok, near_ok, rel, expected in (
            (True, True, True, 0.03, True), (False, True, True, 0.03, False),
            (True, False, True, 0.03, False), (True, True, False, 0.03, False),
            (True, True, True, 0.031, False)):
        ns['_at_score'] = lambda *args: (judge_ok, rel, 0.03, cache_ok)
        ns['_at_judge'] = lambda a, b, tol: (near_ok, 0.001, 0.001) if a == 'out' else (True, 0.03, 0.03)
        assert ns['_df_equiv']({}, 'out', 'base', ('ref', 0.1), (), 1, 'mock') == expected
    # Mock only GPU mechanics, execute the actual candidate's admission loop.
    stream = NS(wait_stream=lambda stream: None)
    torch.cuda = NS(current_device=lambda: 0, current_stream=lambda dev: stream,
                    Stream=lambda **kw: stream, stream=lambda s: nullcontext())
    torch.device = lambda *args: 'cuda:0'
    torch.no_grad = nullcontext
    torch.ones = torch.empty = lambda *args, **kwargs: None
    class Budget(Exception):
        pass
    class DeviceFault(RuntimeError):
        pass
    ns.update(_DF_THREAD=NS(join=lambda **kwargs: None), _DF_MOD=object(),
              _AT_IMPORT_START=0, _DF_BUDGET=220, _at_time=NS(monotonic=lambda: 0),
              _DF_DRAWS=3, _DF_WS={}, _WS={}, _at_check=lambda d: None,
              _workspace=lambda *args: None, _at_drain=lambda d: None,
              _ATBudget=Budget, _DFDeviceFault=DeviceFault)
    load_functions(tree, ['_df_tune'], ns)
    scenarios = 0
    for failures, expected in ((set(), {16, 32}), ({(16, 0)}, {32}),
                               ({(16, 2)}, {32}), ({(32, 2)}, {16}),
                               ({(16, 1), (32, 0)}, set())):
        calls = []
        def draw(B, i, *args):
            calls.append((B, i))
            return (B, i) not in failures
        ns.update(_DF_SELECTED=set(), _DF_DEVICE=None, _df_draw=draw)
        ns['_df_tune']()
        assert ns['_DF_SELECTED'] == expected
        for B in expected:
            assert all((B, i) in calls for i in range(3))
        scenarios += 1
    def timed_out(B, i, *args):
        if B == 32:
            raise Budget()
        return True
    ns.update(_DF_SELECTED=set(), _df_draw=timed_out)
    ns['_df_tune']()
    assert ns['_DF_SELECTED'] == {16}
    def fault(*args):
        raise RuntimeError('mock device failure')
    ns.update(_DF_SELECTED=set(), _df_draw=fault)
    try:
        ns['_df_tune']()
    except DeviceFault:
        pass
    else:
        raise AssertionError('device fault swallowed as fallback')
    return dict(layout_rejections=rejected, admission_scenarios=scenarios + 2,
                timing_gate_cases=8, equivalence_gate_cases=5)


def bandwidth_and_balance():
    result = {}
    for B, us in ((16, 86.48892431597221), (32, 165.73434978928213)):
        # Spec: 2 caches * 2 KV heads * 64 dims * 1 FP8 byte, not 16 Q heads.
        useful = B * 80000 * 2 * 2 * 64
        bw = useful / (us * 1e6)
        result[f'd{B}F'] = dict(nominal_context=80000, kv_bytes=useful,
                                time_us=us, useful_TB_s=bw, peak_fraction=bw / 4.8,
                                peak_floor_us=useful / 4.8e6)
        draws = []
        for draw in range(3):
            seed = 0xAC417AA + 101 * B + 13 + 100003 * (draw + 1)
            rng = random.Random(seed)
            lengths = [rng.randrange(74000, 86001) for _ in range(B)]
            cum = [0]
            for n in lengths:
                cum.append(cum[-1] + n + 1)
            total = cum[-1] + 400 * (B - 1)
            def vmap(v):
                b = max(i for i in range(B) if cum[i] + 400 * i <= v)
                return min(v - b * 400, cum[b + 1])
            work, segments = [], 0
            for c in range(264):
                lo, hi = vmap(c * total // 264), vmap((c + 1) * total // 264)
                count = 0
                for b in range(B):
                    start, end = max(lo, cum[b]), min(hi, cum[b + 1])
                    if start < end:
                        count += (end - start + 63) // 64
                        segments += 1
                work.append(count)
            draws.append(dict(total_cached_tokens=sum(lengths), kv_bytes=256 * sum(lengths),
                              cta_count=264, tile_min=min(work), tile_max=max(work),
                              tile_mean=sum(work) / 264, crossed_request_ctas=segments - 264))
        result[f'd{B}F']['setup_draws'] = draws
    return result


def main():
    text = CANDIDATE.read_text()
    tree = ast.parse(text)
    old, deep = source_checks(text, tree)
    a, b = native_layout(old), native_layout(deep)
    if a['status'] == 'passed':
        assert a['nonflat_sizeof'] == b['nonflat_sizeof']
        assert a['verify_sizeof'] == b['verify_sizeof']
        assert b['shared_residency'] >= 2
    result = dict(candidate=str(CANDIDATE.relative_to(ROOT)),
                  candidate_sha256=hashlib.sha256(text.encode()).hexdigest(),
                  status='CPU/static passed; H200 unmeasured',
                  cuda_device_source_identity='only flat q1 STAGES 3->6',
                  base_layout=a, ring6_layout=b,
                  numerical_model=numerical_model(), policy=policy_checks(tree),
                  bandwidth=bandwidth_and_balance(),
                  toolchain={k: shutil.which(k) for k in ('nvcc', 'ptxas', 'clang++')},
                  gpu_validation='skipped: no local CUDA GPU, torch, NVRTC, nvcc or ptxas')
    RESULTS.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
