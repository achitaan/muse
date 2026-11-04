"""Offline gates for dense_owner_epilogue.py. No CUDA or OKC invocation.

Run with the repository's CPU Torch/NumPy environment. CUDA compilation and
runtime races/performance remain GPU gates, not claims made by this model.
"""
from __future__ import annotations

import ast
import contextlib
import copy
import ctypes.util
import hashlib
import io
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / 'candidates/dense_owner_epilogue.py'
BASE = ROOT / 'board_best/dense_kbv2_scored_1.5886.py'


def definitions(tree, names):
    return ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names], type_ignores=[])


def setup():
    text = CANDIDATE.read_text()
    tree, base = ast.parse(text), ast.parse(BASE.read_text())
    prefix = copy.deepcopy(tree.body[:len(base.body)])
    # The only modification to the incumbent exports its own source/configs.
    for node, old in zip(prefix, base.body):
        if isinstance(node, ast.FunctionDef) and node.name == '_make_vjeux':
            assert 'owner_parts' in ast.unparse(node.body[-1])
            node.body[-1] = copy.deepcopy(old.body[-1])
    assert ast.dump(ast.Module(body=prefix, type_ignores=[])) == ast.dump(base)
    ns = {'_PRUNE': True, '_TABLE': {}, '_DEFAULT': 'chenwainuo'}
    exec(compile(definitions(tree, {'_make_vjeux'}), str(CANDIDATE), 'exec'), ns)
    parts = ns['_make_vjeux']()['owner_parts']
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id.startswith('_OE_') for t in node.targets):
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(CANDIDATE), 'exec'), ns)
    funcs = {n.name for n in tree.body if isinstance(n, ast.FunctionDef) and n.name.startswith('_oe_')}
    # Only the last production wrapper; incumbent definitions are never imported/run.
    exec(compile(definitions(tree, funcs), str(CANDIDATE), 'exec'), ns)
    last = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'kernel_fn'][-1]
    exec(compile(ast.Module(body=[last], type_ignores=[]), str(CANDIDATE), 'exec'), ns)
    ns.update(_dt_torch=torch, _dt_time=time, _K={'vjeux': {'owner_parts': parts}}, _dt_check=lambda d: None)
    return text, tree, ns, parts


def source_gates(ns, parts):
    source = ns['_oe_source'](parts)
    assert '@@' not in source
    assert source.count('launch_owner<') == 4
    assert source.count('named_bar(1, NC)') == 1
    assert 'static_assert(WSK == 2 && CL == 1 && COMP && !HL' in source
    assert 'TOKS * R * sizeof(float)' in source
    begin = '  // ---------------- epilogue: K-warps reduce through shared memory'
    control = parts['sk2'][:parts['sk2'].index(begin)]
    candidate = source[source.index('#define SK2_WAIT'):source.index('  // No ring reuse:')]
    control = control.replace('sk2_kernel', 'owner_kernel')
    control = control.replace('float* red = reinterpret_cast<float*>(gbase);   // reuses the ring once every stage is consumed',
                              'float* red = reinterpret_cast<float*>(ptab_s + pmax) + 32;   // disjoint from the live ring')
    assert candidate.strip() == control.strip()
    assert '*reinterpret_cast<float2*>(red + pub)' in source
    assert 'cvt.rn.f16' not in source
    assert source.count('mma_bf16(acc[') == parts['sk2'].count('mma_bf16(acc[')
    # Save the exact generated translation unit for a caller's offline compiler.
    destination = ROOT / 'results/dense-owner-offline.cu'
    destination.write_text(source)
    return hashlib.sha256(source.encode()).hexdigest()


def configurations(parts):
    rows = []
    for layer, m in (('o', 32), ('unpooler', 16), ('unpooler', 32), ('pooler', 32)):
        cfg = next(c for c in parts['configs'] if c[0] == parts['pick'][layer, m])
        _, _, n, r, nt, nwr, wsk, upw, nacc, cl = cfg
        hl, stages = parts['geom'](cfg, m)
        assert not hl and wsk == 2 and cl == 1 and n % r == 0
        mt = 2 if m == 32 else 1
        nacc = 1 if mt == 2 and stages >= 4 and nt == 4 else nacc
        tu, nw = 768, nwr * wsk
        sst = ((wsk * upw * m * 128 + nw * upw * nt * tu + 1023) // 1024) * 1024
        toks, cp = 16 * mt, r + 1
        nesc = min(stages, 4)
        # pmax=64 as in the original ring-size estimate. Real pmax is logged
        # by the GPU's stream preparation and changes both paths equally.
        base_smem = 1024 + stages * sst + 16 * stages + 32 + nesc * ((toks * cp + 3) & ~3) * 4 + 64 * 8 + 128
        owner_smem = base_smem + toks * r * 4
        assert owner_smem < 232448
        assert (228 * 1024 // base_smem) == (228 * 1024 // owner_smem)
        rows.append(dict(layer=layer, m=m, cfg=cfg[0], mt=mt, nt=nt, nwr=nwr, nacc=nacc, r=r,
                         stages=stages, nesc=nesc, base_smem=base_smem, owner_smem=owner_smem,
                         original_partial_bytes_each_direction=2 * toks * r * 4,
                         owner_partial_bytes_each_direction=toks * r * 4))
    return rows


def folded(acc):
    result = acc[..., 0].copy()
    for c in range(1, acc.shape[-1]):
        result = np.float32(result + acc[..., c])
    return result


def emulate(acc, corr, m, perm=None):
    """The actual lane/register/scratch mapping and FP32 operation sequence."""
    nwr, wsk, mt, nt, lanes, four, nacc = acc.shape
    assert wsk == 2 and lanes == 32 and four == 4
    r, toks = nwr * nt * 8, mt * 16
    partial = folded(acc)
    red = np.full((nwr * mt * nt * 128,), np.nan, dtype=np.float32)
    writes = np.zeros(red.shape, dtype=np.int8)
    expected = np.full((toks, r), np.nan, dtype=np.float32)
    out = np.full((toks, r), np.nan, dtype=np.float32)
    counts = np.zeros((toks, r), dtype=np.int8)
    for wr in range(nwr):
        for wk in range(2):
            for a in range(mt):
                for b in range(nt):
                    for lane in range(32):
                        pub = ((wr * mt + a) * nt + b) * 128 + wk * 64 + lane * 2
                        other = 2 * (1 - wk)
                        red[pub:pub + 2] = partial[wr, wk, a, b, lane, other:other + 2]
                        writes[pub:pub + 2] += 1
    assert np.all(writes == 1) and np.all(np.isfinite(red))
    for wr in range(nwr):
        for a in range(mt):
            for b in range(nt):
                for lane in range(32):
                    g, t4 = lane >> 2, lane & 3
                    row = (wr * nt + b) * 8 + 2 * t4
                    for wk in range(2):
                        tok = a * 16 + g + wk * 8
                        src = ((wr * mt + a) * nt + b) * 128 + (1 - wk) * 64 + lane * 2
                        p = red[src:src + 2]
                        own = partial[wr, wk, a, b, lane, 2 * wk:2 * wk + 2]
                        v = np.zeros(2, dtype=np.float32)
                        for c in range(corr.shape[0]):
                            v = np.float32(v + corr[c, tok, row:row + 2])
                        v = np.float32(v + (own if wk == 0 else p))
                        v = np.float32(v + (p if wk == 0 else own))
                        ref = np.zeros(2, dtype=np.float32)
                        for c in range(corr.shape[0]):
                            ref = np.float32(ref + corr[c, tok, row:row + 2])
                        for w in range(2):
                            ref = np.float32(ref + partial[wr, w, a, b, lane, wk * 2:wk * 2 + 2])
                        col = row if perm is None else perm[row]
                        assert col % 2 == 0
                        expected[tok, col:col + 2] = ref
                        if tok < m:
                            out[tok, col:col + 2] = v
                            counts[tok, col:col + 2] += 1
    assert np.all(counts[:m] == 1) and not np.any(counts[m:])
    assert np.isnan(out[m:]).all()  # output canary: no out-of-bounds tokens
    assert np.array_equal(out[:m].view(np.uint32), expected[:m].view(np.uint32))
    return out[:m]


def epilogue_gates(rows):
    rng = np.random.default_rng(184519)
    cases = 0
    for item in rows:
        mt, nt, nwr, r, nesc = (item[x] for x in ('mt', 'nt', 'nwr', 'r', 'nesc'))
        for nacc in (1, 2, 4):
            shape = (nwr, 2, mt, nt, 32, 4, nacc)
            acc = (rng.standard_normal(shape) * np.exp2(rng.integers(-20, 20, shape))).astype(np.float32)
            corr = (rng.standard_normal((nesc, mt * 16, r)) * 1e-3).astype(np.float32)
            for m in range(1, mt * 16 + 1):
                perm = (rng.permutation(r // 4)[:, None] * 4 + np.arange(4)[None, :]).ravel()
                emulate(acc, corr, m, perm)
                cases += 1
    return cases


def packing_gates(tree):
    factory = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_make_vjeux')
    selected = {'_exponent_window', '_escapes', 'sk2_tileunits', 'sk2_compress', 'sk2_stream'}
    ns = dict(torch=torch, EXPONENT_WINDOW=8, EXPONENT_VALUES=128, MAX_ESCAPE_DENOMINATOR=200,
              BASE4_REPLICATE=0x01010101)
    exec(compile(definitions(factory, selected), str(CANDIDATE), 'exec'), ns)
    gen = torch.Generator().manual_seed(98117)
    w = (torch.randn((64, 8192), generator=gen) / 8192 ** 0.5).bfloat16()
    w[0, 0], w[1, 1], w[2, 2] = 0, -0.0, 2.0 ** -30
    c = ns['sk2_compress'](w)
    assert c is not None and len(c['rows']) > 0
    packed = c['packed'].to(torch.int32)
    t, u = packed.shape[:2]
    low = packed[..., :512].reshape(t, u, 32, 16)
    code = packed[..., 512:].reshape(t, u, 32, 8)
    code = torch.cat((code[..., :4] & 15, code[..., :4] >> 4,
                      code[..., 4:] & 15, code[..., 4:] >> 4), dim=-1)
    high = ((code & 8) << 4) | ((c['base4'] & 255) + (code & 7))
    decoded = (low | (high << 8)).to(torch.int16)
    dense = decoded.reshape(t, u, 8, 4, 4, 2, 2).permute(0, 2, 1, 4, 5, 3, 6).reshape(w.shape)
    dense[c['rows'], c['cols']] = c['wbits'].to(torch.int16)
    assert torch.equal(dense, w.view(torch.int16))
    # The prepared stream's K partition is unchanged and covers every unit once.
    units = ns['sk2_tileunits'](w)
    stream = ns['sk2_stream'](units, 64, 32, 2, 2, 2, 2)
    assert stream.shape == (32, 2, 4, 2, 2, 32, 16)
    for s in (0, 1, 17, 31):
        for cta in (0, 1):
            for warp in range(4):
                for uu in range(2):
                    for nt in range(2):
                        assert torch.equal(stream[s, cta, warp, uu, nt],
                                           units[cta * 4 + (warp // 2) * 2 + nt, s * 4 + (warp % 2) * 2 + uu])
    return dict(weights=w.numel(), escapes=len(c['rows']), exact_bits=True)


def numerical_gates():
    gen = torch.Generator().manual_seed(59917)
    rng = np.random.default_rng(891)
    rows, k = 32, 1024
    w = (torch.randn(rows, k, generator=gen) / k ** 0.5).bfloat16()
    results = []
    for m in (9, 16, 17, 31, 32):
        for dist in ('normal', 'outliers', 'probe'):
            x = torch.randn(m, k, generator=gen).bfloat16()
            if dist == 'outliers':
                x[:, :10] *= 30
            if dist == 'probe':
                x.zero_()
                x[torch.arange(m), torch.arange(m) * 13 % k] = torch.where(torch.arange(m) % 2 == 0, 8., -32.).bfloat16()
            partials = []
            for wk in range(2):
                take = ((torch.arange(k) // 64) % 4) // 2 == wk
                partials.append(x[:, take].float() @ w[:, take].float().t())
            mt = 1 if m <= 16 else 2
            acc = np.zeros((2, 2, mt, 2, 32, 4, 1), dtype=np.float32)
            for wr in range(2):
                for wk in range(2):
                    for a in range(mt):
                        for b in range(2):
                            for lane in range(32):
                                g, t4 = lane >> 2, lane & 3
                                row = (wr * 2 + b) * 8 + 2 * t4
                                for half in range(2):
                                    tok = a * 16 + g + 8 * half
                                    if tok < m:
                                        acc[wr, wk, a, b, lane, 2 * half:2 * half + 2, 0] = partials[wk][tok, row:row + 2].numpy()
            out = torch.from_numpy(emulate(acc, np.zeros((3, mt * 16, rows), dtype=np.float32), m)).bfloat16()
            ref = x.double() @ w.double().t()
            err = float((out.double() - ref).norm() / ref.norm())
            assert err <= (1e-3 if dist == 'probe' else 1e-2)
            if dist == 'probe':
                assert torch.equal(out, ref.bfloat16())
            # The scratch has no numerical dependence on any previous graph replay.
            emulate(acc, np.zeros((3, mt * 16, rows), dtype=np.float32), m)
            results.append(dict(m=m, dist=dist, rel_l2=err))
    return results


def wrapper_gates(ns):
    calls = []

    class Mod:
        def owner_run(self, *args):
            calls.append(args)
            args[2].copy_(args[1].float().sum(1, keepdim=True).expand_as(args[2]).bfloat16())

    cells = set(ns['_OE_CELLS'])
    ns.update(_v6_n=lambda layer, m: 1024 if layer == 'unpooler' and m <= 8 else {'o': 8192, 'unpooler': 8192, 'pooler': 4096}[layer],
              _V6_K={'o': 1024, 'unpooler': 4096, 'pooler': 8192}, _ctx_bind=lambda x: None,
              _oe_parent_kernel_fn=lambda d, s: 'fallback')
    vs = {'sk2': {}, 'dummy': torch.empty(0)}
    for layer, m in cells:
        name = ns['_K']['vjeux']['owner_parts']['pick'][layer, m]
        vs['sk2'][name] = (7, torch.empty(0), torch.empty(0), 4, 5, ns['_v6_n'](layer, m))
    state = dict(owner_cells=cells, owner_mod=Mod(), vjeux=vs)
    for layer, m in cells:
        k, n = ns['_V6_K'][layer], ns['_v6_n'](layer, m)
        x = torch.zeros(m, k, dtype=torch.bfloat16)
        cfg = dict(layer=layer, m=m, n=n, k=k)
        a = ns['kernel_fn'](dict(x=x, config=cfg), state)
        x[:, 0] = 1
        b = ns['kernel_fn'](dict(x=x, config=cfg), state)
        assert a.data_ptr() != b.data_ptr() and torch.equal(a, torch.zeros_like(a)) and torch.equal(b, torch.ones_like(b))
        assert calls[-1][6:9] == (m, n, k)
        assert ns['kernel_fn'](dict(x=x.float(), config=cfg), state) == 'fallback'
        assert ns['kernel_fn'](dict(x=x, config=dict(cfg, n=n + 4)), state) == 'fallback'
        nc = torch.zeros(k, m, dtype=torch.bfloat16).t()
        assert ns['kernel_fn'](dict(x=nc, config=cfg), state) == 'fallback'
        mr = m - 1
        assert ns['kernel_fn'](dict(x=x[:mr], config=dict(cfg, m=mr)), state) == 'fallback'
    assert len(calls) == 2 * len(cells)
    return len(calls)


def policy_gates(ns):
    saved = dict(ns)
    try:
        ns.update(_dt_inputs=lambda m, k, dev, gen: [torch.zeros(m, 1)],
                  _oe_capture=lambda fn: (None, torch.zeros(1)), _dt_close=lambda a, b, tol: True)
        state, weights = {}, {'o': torch.zeros(1)}
        for draws, expected in (([(.974, .974)] * 3, True),
                                ([(.974, .974), (.98, .974)], False),
                                ([(.974, 1.01)], False),
                                ([(.975, .975)] * 3, True),
                                ([(float('nan'), .9)], False)):
            it = iter(draws)
            ns['_dt_abba'] = lambda *args: (10., 9., *next(it))
            with contextlib.redirect_stdout(io.StringIO()):
                result = ns['_oe_measure']('o', 32, state, None, weights, None, None, None, float('inf'))
            assert result is expected
        ns['_dt_close'] = lambda *args: False
        ns['_dt_abba'] = lambda *args: (10., 9., .9, .9)
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                ns['_oe_measure']('o', 32, state, None, weights, None, None, None, float('inf'))
            except RuntimeError as error:
                assert 'numerical mismatch' in str(error)
            else:
                raise AssertionError('numerical failure was admitted')
    finally:
        ns.clear()
        ns.update(saved)
    return 6


def main():
    torch.set_num_threads(4)
    text, tree, ns, parts = setup()
    assert text.startswith('# okc: problem=muse_bf16_proj\n')
    assert b'\r' not in CANDIDATE.read_bytes()
    sha = source_gates(ns, parts)
    rows = configurations(parts)
    report = dict(candidate_sha256=hashlib.sha256(CANDIDATE.read_bytes()).hexdigest(),
                  generated_cuda_sha256=sha, incumbent_ast_preserved=True,
                  configs=rows, exact_epilogue_cases=epilogue_gates(rows),
                  packing=packing_gates(tree), numerical=numerical_gates(),
                  fresh_wrapper_calls=wrapper_gates(ns), policy_cases=policy_gates(ns),
                  cuda_tools={**{name: shutil.which(name) for name in ('nvcc', 'ptxas')},
                              'nvrtc_library': ctypes.util.find_library('nvrtc')},
                  cuda_compile='SKIPPED: no CUDA/NVRTC toolchain on this macOS host',
                  gpu='NOT RUN; no GPU and no OKC check/submit authorized')
    (ROOT / 'results/dense-owner-offline.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
