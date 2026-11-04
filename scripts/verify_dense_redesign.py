"""CPU/static gates for the independent clustered dense restart.

Models are not CUDA compilation, GPU synchronization validation or timing.
Only the explicitly supplied first-check profile is read by the roofline tool.
"""
from __future__ import annotations

import ast
import contextlib
import ctypes.util
import hashlib
import io
import json
from pathlib import Path
import random
import re
import shutil

import numpy as np
import torch

from dense_redesign_roofline import analyze, plan as roof_plan
from dense_redesign_syntax import check as syntax_check

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / 'candidates/dense_redesign.py'
BASE = ROOT / 'board_best/dense_kbv2_scored_1.5886.py'


def setup():
    tree, base = ast.parse(CANDIDATE.read_text()), ast.parse(BASE.read_text())
    assert ast.dump(ast.Module(body=tree.body[:len(base.body)], type_ignores=[])) == ast.dump(base)
    ns = dict(_dt_torch=torch, _dt_check=lambda d: None)
    for node in tree.body[len(base.body):]:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id.startswith('_RD_') for t in node.targets):
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(CANDIDATE), 'exec'), ns)
        if isinstance(node, ast.FunctionDef) and (node.name.startswith('_rd_') or node.name == 'kernel_fn'):
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(CANDIDATE), 'exec'), ns)
    return tree, ns


def source_gates(tree, ns):
    source = ns['_rd_source']()
    assert '@@' not in source and source.count('template <>') == 4
    assert source.count('wgmma.mma_async.sync.aligned.m64n') == 4
    assert 'bf16.bf16' in source and '.f16.f16' not in source
    for tok in (16, 32, 128, 256):
        start = source.index('void mma<%d>' % tok)
        end = source.index('\n}', start)
        body = source[start:end]
        assert body.count('"+f"(d[') == tok // 2
        assert set(int(i) for i in re.findall(r'"\+f"\(d\[(\d+)\]\)', body)) == set(range(tok // 2))
    # Consumer re-arm precedes both the local and remote empty arrivals.
    release = source[source.index('void release_slot'):source.index('int owner(')]
    assert release.index('arm(') < release.index('arrive(') < release.index('arrive_rank0(')
    producer = source[source.index('if (warp == 4)'):source.index('} else {\n    int unit')]
    assert 'arm(' not in producer and 'cluster_sync' not in producer
    assert 'released = final_seq;' in source and 'if (seq - 1 > released)' in source
    assert source.index('__stcg(dst') < source.index('__threadfence();') < source.index('publish_count(counts')
    assert 'atomicExch(counts + nb, 0)' in source
    assert source.count('cluster_sync();') == 2
    assert 'STREAM ? STAGES * ST + 64 : 0' in source
    assert len(re.findall(r'\bat::Tensor\b', ns['_RD_CPP'].split('std::vector')[0])) == 5
    assert ns['_RD_CPP'].split('std::vector')[0].count('int64_t') == 4
    # Timed Python wrappers perform allocation/metadata dispatch/native launch only.
    final = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'kernel_fn'][-1]
    timed = [final, next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_rd_call')]
    forbidden = {'item', 'cpu', 'numpy', 'synchronize', 'double', 'fill_', 'any', 'print', 'open'}
    for node in timed:
        for call in ast.walk(node):
            if isinstance(call, ast.Call):
                name = call.func.attr if isinstance(call.func, ast.Attribute) else call.func.id if isinstance(call.func, ast.Name) else ''
                assert name not in forbidden
    path = ROOT / 'results/dense-redesign-offline.cu'
    path.write_text(source)
    return hashlib.sha256(source.encode()).hexdigest()


def contribution_tasks(g):
    n, k = g['n'], g['k']
    pairs, kc, groups = n // 128, k // 64, g['groups']
    total = pairs * kc
    tasks, coverage = [], np.zeros((pairs, kc), dtype=np.int16)
    for group in range(groups):
        begin = total * group // groups if g['stream'] else group * kc
        end = total * (group + 1) // groups if g['stream'] else (group + 1) * kc
        assert end > begin
        pos = begin
        while pos < end:
            pair = pos // kc
            stop = min(end, (pair + 1) * kc)
            first = ((pair * kc + 1) * groups - 1) // total if g['stream'] else pair
            last = (((pair + 1) * kc) * groups - 1) // total if g['stream'] else pair
            slot, contributors = group - first, last - first + 1
            assert 0 <= slot < contributors <= g['maxparts']
            coverage[pair, pos % kc:stop - pair * kc] += 1
            for rank in range(2):
                tasks.append((pair * 2 + rank, group, slot, contributors, pos % kc * 64, (stop - pair * kc) * 64))
            pos = stop
    assert np.all(coverage == 1)
    for nb in range(n // 64):
        mine = [t for t in tasks if t[0] == nb]
        assert len(mine) == mine[0][3]
        assert sorted(t[2] for t in mine) == list(range(len(mine)))
        assert len(set((t[1], t[0]) for t in mine)) == len(mine)
    return tasks, coverage


def ownership_gates(ns):
    records = []
    shape = {'qkv_gate': (2304, 8192), 'o': (8192, 1024), 'pooler': (4096, 8192), 'unpooler': (8192, 4096)}
    for layer, m in ns['_RD_CELLS']:
        n, k = shape[layer]
        g = ns['_rd_geometry'](n, k, m)
        modeled = roof_plan(layer, m)
        for key in ('native', 'stream', 'maxparts'):
            assert g[key] == modeled[key]
        assert g['groups'] == modeled['clusters']
        tasks, cov = contribution_tasks(g)
        assert modeled['smem_bytes'] <= 232448
        assert modeled['weight_bytes'] == cov.size * 2 * 64 * 64 * 2
        assert all(t[5] > t[4] for t in tasks)
        records.append(dict(layer=layer, m=m, mode='stream' if g['stream'] else 'output',
                            clusters=g['groups'], parts=g['maxparts'], partial_tasks=len(tasks),
                            weight_stage_pairs=cov.size, max_dynamic_shared=modeled['smem_bytes']))
    # General boundary geometries, including sub-three-stage Stream-K ranges.
    for n in (128, 256, 384, 2304, 4096, 8192):
        for k in (64, 128, 192, 1024, 4096, 8192):
            contribution_tasks(ns['_rd_geometry'](n, k, 129))
    return records


def layout_gates():
    cases = 0
    rng = np.random.default_rng(12199)
    for tok in (16, 32, 128, 256):
        native = np.full((tok, 68), np.nan, dtype=np.float32)
        writes = np.zeros((tok, 68), dtype=np.int16)
        for tid in range(128):
            warp, lane = tid // 32, tid % 32
            row, t = warp * 16 + lane // 4, lane % 4
            for j in range(tok // 8):
                tt = j * 8 + 2 * t
                for rr, tr in ((row, tt), (row, tt + 1), (row + 8, tt), (row + 8, tt + 1)):
                    native[tr, rr] = tr * 64 + rr
                    writes[tr, rr] += 1
        assert np.all(writes[:, :64] == 1) and not np.any(writes[:, 64:])
        # All M tails and vector-store alignment/coverage; no output past M.
        for m in range(1, tok + 1):
            out = np.full((tok, 64), np.nan, dtype=np.float32)
            count = np.zeros((tok, 64), dtype=np.int8)
            for tid in range(128):
                for idx in range(tid, m * 16, 128):
                    tt, rr = idx // 16, idx % 16 * 4
                    assert (tt * 68 + rr) % 4 == 0
                    out[tt, rr:rr + 4] = native[tt, rr:rr + 4]
                    count[tt, rr:rr + 4] += 1
            assert np.all(count[:m] == 1) and not np.any(count[m:])
            assert np.array_equal(out[:m], np.arange(m * 64, dtype=np.float32).reshape(m, 64))
            cases += 1
        # Raw BF16 TMA SW128 addressing roundtrip, for both operands.
        for rows in (64, tok):
            bits = rng.integers(0, 65536, (rows, 64), dtype=np.uint16)
            packed = np.zeros((rows * 64,), dtype=np.uint16)
            seen = set()
            for row in range(rows):
                for col in range(64):
                    addr = row * 64 + (((col // 8) ^ (row & 7)) * 8) + col % 8
                    assert addr not in seen
                    seen.add(addr)
                    packed[addr] = bits[row, col]
            restored = np.empty_like(bits)
            for row in range(rows):
                for col in range(64):
                    addr = row * 64 + (((col // 8) ^ (row & 7)) * 8) + col % 8
                    restored[row, col] = packed[addr]
            assert np.array_equal(bits, restored)
    return cases


def credit_schedule(length, start, kc, seed):
    """Two independent producers/TMA completions/async WG consumers.

    Full phases are armed before empty credits become visible. A returned
    slot must have no outstanding async MMA read. Remote arrivals are delayed
    independently; each destination of a multicast completes independently.
    """
    rng = random.Random(seed)
    armed = [[s if s < length else None for s in range(3)] for _ in range(2)]
    loaded = [[set() for _ in range(3)] for _ in range(2)]
    produced, seqs = [0, 0], [0, 0]
    pc, current = ['issue', 'issue'], [-1, -1]
    groups = [[], []]
    done = [set(), set()]
    released = [set(), set()]
    credits0, credits1 = {}, set()
    requests, remote = set(), set()
    steps = 0

    def release(rank, seq):
        assert seq in done[rank] and seq not in released[rank]
        assert armed[rank][seq % 3] == seq
        released[rank].add(seq)
        armed[rank][seq % 3] = seq + 3 if seq + 3 < length else None
        loaded[rank][seq % 3] = set()
        if rank == 0:
            credits0[seq] = credits0.get(seq, 0) | 1
        else:
            credits1.add(seq)
            remote.add(seq)

    while not (all(p == length for p in produced) and all(p == 'end' for p in pc) and not requests and not remote):
        events = []
        for rank in range(2):
            seq = produced[rank]
            credit = seq < 3 or (credits0.get(seq - 3, 0) == 3 if rank == 0 else seq - 3 in credits1)
            if seq < length and credit:
                events.append(('produce', rank))
            if pc[rank] == 'issue':
                seq = seqs[rank]
                if seq == length:
                    events.append(('end', rank))
                elif loaded[rank][seq % 3] == {('W', seq), ('X', seq)}:
                    events.append(('issue', rank))
            if pc[rank] == 'wait1' and sum(s not in done[rank] for s in groups[rank]) <= 1:
                events.append(('wait1', rank))
            if pc[rank] == 'wait0' and all(s in done[rank] for s in groups[rank]):
                events.append(('wait0', rank))
            if pc[rank] == 'epilogue':
                events.append(('epilogue', rank))
            unfinished = [s for s in groups[rank] if s not in done[rank]]
            if unfinished:
                events.append(('mma_done', rank, unfinished[0]))
        events += [('copy', *r) for r in requests]
        events += [('remote', s) for s in remote]
        assert events, ('deadlock', length, start, kc, seed, pc, produced, seqs)
        event = rng.choice(events)
        kind, args = event[0], event[1:]
        if kind == 'produce':
            rank = args[0]
            seq = produced[rank]
            assert armed[rank][seq % 3] == seq
            requests.add(('W', rank, seq))
            if rank == 0:
                assert all(armed[r][seq % 3] == seq for r in range(2))
                requests.add(('X', 0, seq)); requests.add(('X', 1, seq))
            produced[rank] += 1
        elif kind == 'copy':
            which, rank, seq = args
            assert armed[rank][seq % 3] == seq
            assert not any(s % 3 == seq % 3 and s != seq and s not in done[rank] for s in groups[rank])
            loaded[rank][seq % 3].add((which, seq))
            requests.remove((which, rank, seq))
        elif kind == 'remote':
            seq = args[0]
            assert seq in released[1]
            credits0[seq] = credits0.get(seq, 0) | 2
            remote.remove(seq)
        elif kind == 'issue':
            rank = args[0]
            seq = seqs[rank]
            assert armed[rank][seq % 3] == seq
            groups[rank].append(seq)
            current[rank], pc[rank] = seq, 'wait1'
        elif kind == 'mma_done':
            rank, seq = args
            done[rank].add(seq)
        elif kind == 'wait1':
            rank = args[0]
            seq = current[rank]
            if seq > 0 and seq - 1 not in released[rank]:
                release(rank, seq - 1)
            is_boundary = seq + 1 == length or (start + seq) // kc != (start + seq + 1) // kc
            if is_boundary:
                pc[rank] = 'wait0'
            else:
                seqs[rank], pc[rank] = seq + 1, 'issue'
        elif kind == 'wait0':
            rank = args[0]
            seq = current[rank]
            if seq not in released[rank]:
                release(rank, seq)
            pc[rank] = 'epilogue'
        elif kind == 'epilogue':
            rank = args[0]
            seqs[rank], pc[rank] = current[rank] + 1, 'issue'
        elif kind == 'end':
            pc[args[0]] = 'end'
        steps += 1
        assert steps < length * 40 + 100
    assert all(len(r) == length for r in released)
    assert all(d == set(range(length)) for d in done)
    return steps


def barrier_gates(ns):
    count, steps = 0, 0
    for seed in range(1200):
        length = (1, 2, 3, 4, 5, 16, 34, 35, 62, 63)[seed % 10]
        kc = (2, 16, 64, 128)[seed % 4]
        start = (seed * 11) % kc
        steps += credit_schedule(length, start, kc, seed * 1337 + 9)
        count += 1
    # Also exercise every actual cluster range in the four distinct scored
    # K/N geometries, including all real Stream-K row-pair transitions.
    for n, k in ((2304, 8192), (4096, 8192), (8192, 4096), (8192, 1024)):
        g = ns['_rd_geometry'](n, k, 256)
        kc, total = k // 64, (n // 128) * (k // 64)
        for group in range(g['groups']):
            begin = total * group // g['groups'] if g['stream'] else group * kc
            end = total * (group + 1) // g['groups'] if g['stream'] else (group + 1) * kc
            steps += credit_schedule(end - begin, begin % kc, kc, group * 715 + n)
            count += 1
    return dict(schedules=count, asynchronous_events=steps)


def replay_model(g, x, w, order_seed, counters=None, drop=False):
    tasks, _ = contribution_tasks(g)
    m, n = x.shape[0], g['n']
    parts = np.full((n // 64, g['maxparts'], m, 64), np.nan, dtype=np.float32)
    counts = np.zeros(n // 64, dtype=np.int32) if counters is None else counters
    assert not np.any(counts)
    outputs = np.full((m, n), np.nan, dtype=np.float32)
    written = np.zeros(n // 64, dtype=np.int8)
    random.Random(order_seed).shuffle(tasks)
    if drop:
        tasks = tasks[:-1]
    for nb, group, slot, nc, k0, k1 in tasks:
        value = x[:, k0:k1].float() @ w[nb * 64:(nb + 1) * 64, k0:k1].float().t()
        if g['stream']:
            parts[nb, slot] = value.numpy()
            old = counts[nb]
            counts[nb] += 1
            assert old < nc
            if old == nc - 1:
                assert np.isfinite(parts[nb, :nc]).all()
                value = np.zeros((m, 64), dtype=np.float32)
                for s in range(nc):
                    value = np.float32(value + parts[nb, s])
                outputs[:, nb * 64:(nb + 1) * 64] = value
                counts[nb] = 0
                written[nb] += 1
        else:
            outputs[:, nb * 64:(nb + 1) * 64] = value.numpy()
            written[nb] += 1
    assert np.all(written == 1) and not np.any(counts) and np.isfinite(outputs).all()
    return torch.from_numpy(outputs).bfloat16(), counts


def numerical_gates(ns):
    gen = torch.Generator().manual_seed(195911)
    cases = []
    for n, k, m in ((256, 128, 128), (384, 192, 256), (256, 1024, 127), (384, 1024, 129),
                    (512, 256, 255), (8192, 64, 16), (8192, 64, 32), (8192, 64, 128), (8192, 128, 256)):
        g = ns['_rd_geometry'](n, k, m)
        w = (torch.randn(n, k, generator=gen) / k ** .5).bfloat16()
        counters = None
        for dist in ('normal', 'outliers', 'probes', 'zero'):
            x = torch.randn(m, k, generator=gen)
            if dist == 'outliers':
                x[:, ::97] *= 30
            if dist in ('probes', 'zero'):
                x.zero_()
            if dist == 'probes':
                rows = torch.arange(m)
                cols = (rows * 63) % k
                x[rows, cols] = torch.where(rows % 2 == 0, 8., -32.)
            x = x.bfloat16()
            out, counters = replay_model(g, x, w, m + k, counters)
            again, counters = replay_model(g, x, w, k * 17 + m, counters)
            assert torch.equal(out, again)
            ref = x.double() @ w.double().t()
            rn = ref.norm(dim=-1)
            denom = torch.maximum(rn, .1 * rn.mean()).clamp_min(1e-30)
            err = float((out.double() - ref).norm() / ref.norm().clamp_min(1e-30))
            row = float(((out.double() - ref).norm(dim=-1) / denom).max())
            bound = 1e-3 if dist == 'probes' else 1e-2
            assert err <= bound and row <= (bound if dist == 'probes' else 5 * bound)
            if dist == 'probes':
                assert torch.equal(out, ref.bfloat16())
            cases.append(dict(n=n, k=k, m=m, dist=dist, stream=g['stream'], relative_l2=err, row_error=row))
        if g['stream']:
            try:
                replay_model(g, x, w, 31, drop=True)
            except AssertionError:
                pass
            else:
                raise AssertionError('missing contribution was not detected')
    return cases


def wrapper_policy_gates(ns):
    calls = []

    class Mod:
        def rd_run(self, *args):
            calls.append(args)
            args[2].copy_(args[1].float().sum(1, keepdim=True).expand_as(args[2]).bfloat16())

    ns.update(_ctx_bind=lambda x: None, _rd_parent_kernel_fn=lambda d, s: 'fallback')
    mod, state = Mod(), dict(restart={}, restart_mod=Mod())
    for n, m in ((8192, 16), (8192, 32), (8192, 128), (8192, 256), (2304, 128), (2304, 256)):
        layer = 'qkv_gate' if n == 2304 else 'o'
        g = ns['_rd_geometry'](n, 64, m)
        g.update(w=torch.empty(0), workspace=torch.empty(0), counts=torch.empty(0, dtype=torch.int32))
        state['restart'][(layer, m)] = g
        x = torch.zeros(m, 64, dtype=torch.bfloat16)
        cfg = dict(layer=layer, m=m, n=n, k=64)
        a = ns['kernel_fn'](dict(x=x, config=cfg), state)
        x[:, 0] = 1
        b = ns['kernel_fn'](dict(x=x, config=cfg), state)
        assert a.data_ptr() != b.data_ptr() and not bool(a.any()) and bool((b == 1).all())
        assert len(calls[-1]) == 9 and calls[-1][5:] == (g['native'], g['groups'], g['maxparts'], g['stream'])
        for bad in (dict(x=x.float(), config=cfg), dict(x=x, config=dict(cfg, n=n + 128)),
                    dict(x=x, config=dict(cfg, k=128)), dict(x=x[:-1], config=dict(cfg, m=m-1)),
                    dict(x=torch.zeros(m, 128, dtype=torch.bfloat16)[:, ::2], config=cfg)):
            assert ns['kernel_fn'](bad, state) == 'fallback'
    assert len(calls) == 12
    saved = dict(ns)
    try:
        ns.update(_dt_inputs=lambda *args: [torch.zeros(1)], _rd_capture=lambda fn: (None, torch.zeros(1)),
                  _dt_close=lambda *args: True)
        p = dict(k=64, n=128, counts=torch.zeros(1, dtype=torch.int32))
        for ratios, expected in (([(.974, .974)] * 3, True), ([(.975, .975)] * 3, True),
                                 ([(.974, .974), (.976, .974)], False), ([(.974, .98)], False),
                                 ([(float('nan'), .9)], False)):
            it = iter(ratios)
            ns['_dt_abba'] = lambda *args: (10., 9., *next(it))
            with contextlib.redirect_stdout(io.StringIO()):
                assert ns['_rd_measure']('o', 16, p, None, {}, {'o': torch.zeros(1)}, None, None, None, float('inf')) is expected
        ns['_dt_close'] = lambda *args: False
        ns['_dt_abba'] = lambda *args: (10., 9., .9, .9)
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                ns['_rd_measure']('o', 16, p, None, {}, {'o': torch.zeros(1)}, None, None, None, float('inf'))
            except RuntimeError:
                pass
            else:
                raise AssertionError('bad numerical result admitted')
    finally:
        ns.clear(); ns.update(saved)
    return dict(fresh_calls=12, fallbacks=30, admission_cases=6)


def main():
    torch.set_num_threads(4)
    assert b'\r' not in CANDIDATE.read_bytes()
    tree, ns = setup()
    roof = analyze()
    assert roof['build_gate_passed']
    report = dict(candidate_sha256=hashlib.sha256(CANDIDATE.read_bytes()).hexdigest(),
                  generated_cuda_sha256=source_gates(tree, ns), incumbent_ast_preserved=True,
                  cpp_syntax=syntax_check(ns['_rd_source'](), ROOT / 'results/dense-redesign-syntax.cpp'),
                  roofline_board_gains={m['name']: m['board_gain'] for m in roof['models']},
                  ownership=ownership_gates(ns), native_tail_layouts=layout_gates(),
                  credit_protocol=barrier_gates(ns), numerical=numerical_gates(ns),
                  wrapper_policy=wrapper_policy_gates(ns),
                  cuda_tools={**{name: shutil.which(name) for name in ('nvcc', 'ptxas')},
                              'nvrtc_library': ctypes.util.find_library('nvrtc')},
                  cuda_compile='SKIPPED: no CUDA/NVRTC toolchain on this macOS host',
                  gpu='NOT RUN: no GPU or OKC calls authorized')
    (ROOT / 'results/dense-redesign-offline.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
