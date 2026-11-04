"""Offline verification of the independent M=1 rotated-sign cgemv candidate.

No GPU, network, OKC, or full submission import. The CPU C++ decoder test
compiles the actual two new CUDA expressions, with a CPU funnel-shift model.
It does not validate GPU lowering or instruction counts.
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
import subprocess

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / 'candidates/dense_cg_rotcodes.py'
BASE = ROOT / 'board_best/dense_kbv2_scored_1.5886.py'


def defs(tree, names):
    return ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names], type_ignores=[])


def setup():
    tree, base = ast.parse(CANDIDATE.read_text()), ast.parse(BASE.read_text())
    prefix = copy.deepcopy(tree.body[:len(base.body)])
    for node, old in zip(prefix, base.body):
        if isinstance(node, ast.FunctionDef) and node.name == '_make_vjeux':
            assert 'rot_parts' in ast.unparse(node.body[-1])
            node.body[-1] = copy.deepcopy(old.body[-1])
    assert ast.dump(ast.Module(body=prefix, type_ignores=[])) == ast.dump(base)
    ns = dict(_PRUNE=True, _TABLE={}, _DEFAULT='chenwainuo')
    exec(compile(defs(tree, {'_make_vjeux'}), str(CANDIDATE), 'exec'), ns)
    parts = ns['_make_vjeux']()['rot_parts']
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id.startswith('_RC_') for t in node.targets):
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(CANDIDATE), 'exec'), ns)
    names = {n.name for n in tree.body if isinstance(n, ast.FunctionDef) and n.name.startswith('_rc_')}
    exec(compile(defs(tree, names), str(CANDIDATE), 'exec'), ns)
    node = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'kernel_fn'][-1]
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(CANDIDATE), 'exec'), ns)
    ns.update(_dt_torch=torch, _K={'vjeux': {'rot_parts': parts}}, _dt_check=lambda deadline: None)
    return tree, ns, parts


def source_gates(ns, parts):
    source = ns['_rc_source'](parts)
    assert '@@' not in source and 'cg_hib(' not in source
    assert source.count('<<<') == 4
    assert source.count('rot_cgemv_kernel<2,') == 4
    begin = parts['cg'].index('template <int ROWS, int NTH, int K>\n__global__ void __launch_bounds__(NTH + 32) cgemv_kernel')
    end = parts['cg'].index('// BF16 GEMV (the unpooler slice', begin)
    expected = parts['cg'][begin:end]
    candidate = source[source.index('template <int ROWS, int NTH, int K>'):]
    candidate = candidate.replace(ns['_RC_NEW_DECODE'], ns['_RC_OLD_DECODE'])
    candidate = candidate.replace('rot_cgemv_kernel', 'cgemv_kernel').replace('void rot_cgemv(', 'void cgemv(')
    assert candidate.rstrip() == expected.rstrip()  # loads, FMA order, escape and epilogue literally unchanged
    (ROOT / 'results/dense-cg-rotcodes-offline.cu').write_text(source)
    return hashlib.sha256(source.encode()).hexdigest()


def rotate(c):
    c = np.asarray(c, dtype=np.uint32)
    h = c & np.uint32(0x80808080)
    return ((c & np.uint32(0x77777777)) | ((c & np.uint32(0x08080808)) << np.uint32(4))
            | (h << np.uint32(4)) | (h >> np.uint32(28)))


def inverse(c):
    c = np.asarray(c, dtype=np.uint32)
    h = c & np.uint32(0x08080808)
    return ((c & np.uint32(0x77777777)) | ((c & np.uint32(0x80808080)) >> np.uint32(4))
            | (h >> np.uint32(4)) | (h << np.uint32(28)))


def decode_high(c, base, rotated):
    b = np.uint32(base * 0x01010101)
    if rotated:
        hi = (c & np.uint32(0x87878787)) + b
        other = (c >> np.uint32(4)) | (c << np.uint32(28))
        lo = (other & np.uint32(0x87878787)) + b
    else:
        hi = ((c & np.uint32(0x07070707)) + b) | ((c << np.uint32(4)) & np.uint32(0x80808080))
        lo = (((c >> np.uint32(4)) & np.uint32(0x07070707)) + b) | (c & np.uint32(0x80808080))
    return np.stack([hi, lo], axis=-1)


def exhaustive_gates(ns):
    rng = np.random.default_rng(512911)
    words = rng.integers(0, 2**32, 65536, dtype=np.uint32)
    words = np.concatenate((np.array([0, 0xffffffff], dtype=np.uint32),
                            np.array([1 << i for i in range(32)], dtype=np.uint32), words))
    rot = rotate(words)
    assert np.array_equal(inverse(rot), words)
    basis = rotate(np.array([1 << i for i in range(32)], dtype=np.uint32))
    assert len(set(int(v) for v in basis)) == 32 and all(int(v).bit_count() == 1 for v in basis)
    assert np.array_equal(ns['_rc_codes'](torch.from_numpy(words.copy().view(np.uint8).reshape(-1, 4))).numpy(),
                          rot.view(np.uint8).reshape(-1, 4))
    checked = 0
    # Every possible BF16 payload, for every legal eight-exponent window base.
    bits = np.arange(65536, dtype=np.uint16).reshape(-1, 8)
    high = bits.astype(np.uint32) >> 8
    exponent = high & 127
    signs = (high >> 7) << 3
    for base in range(121):
        codes = signs | np.clip(exponent.astype(np.int32) - base, 0, 7).astype(np.uint32)
        byte = codes[:, :4] | (codes[:, 4:] << np.uint32(4))
        c = np.zeros((len(byte),), dtype=np.uint32)
        for i in range(4):
            c |= byte[:, i] << np.uint32(8 * i)
        assert np.array_equal(decode_high(c, base, False), decode_high(rotate(c), base, True))
        assert np.array_equal(inverse(rotate(c)), c)
        assert np.array_equal(decode_high(words, base, False), decode_high(rot, base, True))
        checked += bits.size
    for bad in (torch.zeros(4, 6, dtype=torch.uint8), torch.zeros(4, 4), torch.zeros(4, 8, dtype=torch.uint8)[:, ::2]):
        try:
            ns['_rc_codes'](bad)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid code metadata admitted')
    return dict(all_bf16_payload_window_cases=checked, random_code_word_windows=len(words) * 121,
                sign_bit_basis_bijection=True, actual_torch_word_permutations=len(words), invalid_metadata_rejections=3)


def host_cpp_gate(ns, parts):
    compiler = shutil.which('clang++') or shutil.which('g++')
    if compiler is None:
        return dict(result='SKIPPED: no CPU C++ compiler')
    source = r'''#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
uint32_t __funnelshift_r(uint32_t lo, uint32_t hi, unsigned s) {
  return uint32_t(((uint64_t(hi) << 32) | lo) >> (s & 31));
}
'''
    hb = parts['cg'][parts['cg'].index('__device__ __forceinline__ uint32_t cg_hib'):]
    hb = hb[:hb.index('template <int ROWS')]
    source += hb.replace('__device__ __forceinline__', 'inline')
    source += '\nstd::array<uint32_t, 2> original(uint32_t c, uint32_t base4) {\n'
    source += 'uint32_t cws[1] = {c}; int g = 0;\n' + ns['_RC_OLD_DECODE'] + '\nreturn {ha, hb}; }\n'
    source += '\nstd::array<uint32_t, 2> candidate(uint32_t c, uint32_t base4) {\n'
    source += 'uint32_t cws[1] = {c}; int g = 0;\n' + ns['_RC_NEW_DECODE'] + '\nreturn {ha, hb}; }\n'
    source += r'''
uint32_t rot(uint32_t c) {
  uint32_t h = c & 0x80808080u;
  return (c & 0x77777777u) | ((c & 0x08080808u) << 4) | (h << 4) | (h >> 28);
}
uint32_t unrot(uint32_t c) {
  uint32_t h = c & 0x08080808u;
  return (c & 0x77777777u) | ((c & 0x80808080u) >> 4) | (h >> 4) | (h << 28);
}
int main() {
  uint32_t seed = 0x95117821u;
  unsigned long long count = 0;
  for (unsigned b = 0; b <= 120; ++b) {
    for (unsigned i = 0; i < 65536; ++i) {
      seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
      uint32_t c = i < 32 ? 1u << i : seed;
      if (unrot(rot(c)) != c || original(c,b * 0x01010101u) != candidate(rot(c),b * 0x01010101u))
        throw std::runtime_error("compiled decoder mismatch");
      ++count;
    }
  }
  std::cout << "PASS " << count << " compiled word/window cases\n";
}
'''
    path = ROOT / 'results/dense-cg-rotcodes-host.cpp'
    binary = ROOT / 'results/.dense-cg-rotcodes-host'
    path.write_text(source)
    command = [compiler, '-O2', '-std=c++20', str(path), '-o', str(binary)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    output = subprocess.run([str(binary)], check=True, capture_output=True, text=True).stdout.strip()
    binary.unlink()
    assert output == 'PASS 7929856 compiled word/window cases'
    return dict(result=output, command=command)


def packing_functions(tree):
    factory = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_make_vjeux')
    ns = dict(torch=torch, EXPONENT_WINDOW=8, EXPONENT_VALUES=128, MAX_ESCAPE_DENOMINATOR=200,
              BASE4_REPLICATE=0x01010101)
    exec(compile(defs(factory, {'_exponent_window', '_escapes', 'cg_compress'}), str(CANDIDATE), 'exec'), ns)
    return ns['cg_compress']


def decoded(cg, codes, rotated):
    n, k = cg['L'].shape
    c = codes.numpy().copy().view(np.uint32).reshape(n, k // 8)
    high = decode_high(c, cg['base4'] & 255, rotated)
    h = np.empty((n, k // 8, 8), dtype=np.uint32)
    for i in range(4):
        h[..., i] = (high[..., 0] >> np.uint32(8 * i)) & 255
        h[..., i + 4] = (high[..., 1] >> np.uint32(8 * i)) & 255
    return ((h.reshape(n, k) << 24) | (cg['L'].numpy().astype(np.uint32) << 16)).view(np.float32)


def butterfly(v):
    for offset in (16, 8, 4, 2, 1):
        v = np.float32(v + v[..., np.arange(32) ^ offset])
    return v[..., 0]


def gemv_model(wdec, cg, x, nth):
    n, k = wdec.shape
    xf = x.float().numpy().reshape(-1)
    acc = np.zeros((n, nth), dtype=np.float32)
    for j in range(k // (16 * nth)):
        for e in range(16):
            cols = j * 16 * nth + np.arange(nth) * 16 + e
            acc = (acc.astype(np.float64) + wdec[:, cols].astype(np.float64) * xf[cols]).astype(np.float32)
    math = butterfly(acc.reshape(n, nth // 32, 32))
    corr = np.zeros((n // 2, 2, 32), dtype=np.float32)
    tabs = cg['tab'].numpy()
    for cta in range(n // 2):
        for i in range(1, int(tabs[cta, 0, 0]) + 1):
            en, raw = (int(v) for v in tabs[cta, i])
            row, col = en >> 16, en & 65535
            true = np.array([np.uint32(raw << 16)], dtype=np.uint32).view(np.float32)[0]
            delta = np.float32(true - wdec[cta * 2 + row, col])
            c = np.float32(delta * xf[col])
            corr[cta, row, i % 32] = np.float32(corr[cta, row, i % 32] + c)
    correction = butterfly(corr).reshape(n)
    result = np.zeros((n,), dtype=np.float32)
    for warp in range(nth // 32):
        result = np.float32(result + math[:, warp])
    result = np.float32(result + correction)
    return torch.from_numpy(result).bfloat16()


def numerical_gates(tree, ns):
    pack = packing_functions(tree)
    gen = torch.Generator().manual_seed(551881)
    records = []
    roundtrip_weights = escapes = 0
    for k, nth in ((1024, 64), (4096, 128), (8192, 256), (8192, 512)):
        w = (torch.randn(8, k, generator=gen) / k ** .5).bfloat16()
        w[0, 0], w[1, 3], w[2, 4], w[3, 7], w[4, -1] = 0., -0., 2.**-25, -2.**-24, 2.**-24
        cg = pack(w, 2)
        assert cg is not None
        codes = ns['_rc_codes'](cg['C'])
        a, b = decoded(cg, cg['C'], False), decoded(cg, codes, True)
        assert np.array_equal(a.view(np.uint32), b.view(np.uint32))
        # Apply each stored original BF16 payload to prove a lossless roundtrip.
        restored = (b.view(np.uint32) >> 16).astype(np.uint16)
        tabs = cg['tab'].numpy()
        for cta in range(len(tabs)):
            for en, raw in tabs[cta, 1:int(tabs[cta, 0, 0]) + 1]:
                restored[cta * 2 + (int(en) >> 16), int(en) & 65535] = int(raw)
                escapes += 1
        assert np.array_equal(restored, w.view(torch.int16).numpy().view(np.uint16))
        roundtrip_weights += w.numel()
        for dist in ('normal', 'outliers', 'zero', 'probe0', 'probe3', 'probe4', 'probe7', 'probeLast'):
            x = torch.randn(1, k, generator=gen).bfloat16()
            if dist == 'outliers':
                x[:, ::97] *= 30
            if dist == 'zero' or dist.startswith('probe'):
                x.zero_()
            if dist.startswith('probe'):
                col = k - 1 if dist == 'probeLast' else int(dist[5:])
                x[0, col] = -32.0 if col % 2 else 8.0
            old, new = gemv_model(a, cg, x, nth), gemv_model(b, cg, x, nth)
            assert torch.equal(old, new)
            ref = (x.double() @ w.double().t()).reshape(-1)
            err = float((new.double() - ref).norm() / ref.norm().clamp_min(1e-30))
            assert err <= (1e-3 if dist.startswith('probe') else 1e-2)
            if dist.startswith('probe'):
                assert torch.equal(new, ref.bfloat16())
            records.append(dict(k=k, nth=nth, dist=dist, relative_l2=err))
    return dict(roundtrip_weights=roundtrip_weights, escapes=escapes, cases=records)


def wrapper_gates(ns, parts):
    calls = []

    class Mod:
        def rot_cgemv(self, *args):
            calls.append(('new', args))
            args[5].copy_(args[2].float().sum().expand_as(args[5]).bfloat16())

        def cgemv(self, *args):
            calls.append(('control', args))
            args[5].copy_(args[2].float().sum().expand_as(args[5]).bfloat16())

    mod = Mod()
    parts['ext'] = lambda: mod
    ns.update(_ctx_bind=lambda x: None, _rc_parent_kernel_fn=lambda d, s: 'fallback')
    state = dict(rotcg={}, rotcg_mod=mod)
    shapes = {'pooler': (4096, 8192), 'qkv_gate': (2304, 8192), 'o': (8192, 1024), 'unpooler': (1024, 4096)}
    for layer, (n, k) in shapes.items():
        cg = dict(L=torch.empty(0), C=torch.empty(0), tab=torch.empty(0), pmax=2, base4=123)
        rows, nth = parts['configs'][layer]
        plan = dict(cg=cg, codes=torch.ones(1), rows=rows, nth=nth, n=n, k=k)
        state['rotcg'][layer] = plan
        x = torch.zeros(1, k, dtype=torch.bfloat16)
        cfg = dict(layer=layer, m=1, n=n, k=k)
        data = dict(x=x, config=cfg)
        old = ns['kernel_fn'](data, state)
        x[0, 0] = 1
        new = ns['kernel_fn'](data, state)
        assert old.data_ptr() != new.data_ptr() and not bool(old.any()) and bool((new == 1).all())
        assert calls[-1][1][1] is plan['codes']
        assert calls[-1][1][-2:] == (rows, nth)
        ns['_rc_call'](data, plan, mod, control=True)
        assert calls[-1][1][1] is cg['C']
        for bad in (dict(x=x.float(), config=cfg), dict(x=x, config=dict(cfg, m=2)),
                    dict(x=x, config=dict(cfg, k=k + 8)), dict(x=x, config=dict(cfg, n=n + 2)),
                    dict(x=torch.zeros(1, 2 * k, dtype=torch.bfloat16)[:, ::2], config=cfg)):
            assert ns['kernel_fn'](bad, state) == 'fallback'
    return dict(fresh_output_pairs=len(shapes), control_calls=len(shapes), fallback_cases=5 * len(shapes))


def policy_gates(ns):
    saved = dict(ns)
    try:
        ns.update(_dt_inputs=lambda *args: [torch.zeros(1, 1)], _v6_n=lambda layer, m: 1,
                  _rc_capture=lambda fn: (None, torch.zeros(1)), _dt_close=lambda *args: True)
        weights = {'o': torch.zeros(1, 1)}
        for draws, expected in (([(.974, .974)] * 3, True), ([(.975, .975)] * 3, True),
                                ([(.974, .974), (.976, .974)], False), ([(.974, .98)], False),
                                ([(float('nan'), .9)], False)):
            it = iter(draws)
            ns['_dt_abba'] = lambda *args: (10., 9., *next(it))
            with contextlib.redirect_stdout(io.StringIO()):
                assert ns['_rc_measure']('o', {}, None, {}, weights, None, None, None, float('inf')) is expected
        ns['_dt_close'] = lambda *args: False
        ns['_dt_abba'] = lambda *args: (10., 9., .9, .9)
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                ns['_rc_measure']('o', {}, None, {}, weights, None, None, None, float('inf'))
            except RuntimeError as error:
                assert 'numerical mismatch' in str(error)
            else:
                raise AssertionError('incorrect outputs admitted')
    finally:
        ns.clear()
        ns.update(saved)
    return 6


def main():
    torch.set_num_threads(4)
    assert b'\r' not in CANDIDATE.read_bytes()
    assert CANDIDATE.read_text().startswith('# okc: problem=muse_bf16_proj\n')
    tree, ns, parts = setup()
    report = dict(candidate_sha256=hashlib.sha256(CANDIDATE.read_bytes()).hexdigest(),
                  generated_cuda_sha256=source_gates(ns, parts), incumbent_ast_and_cg_core_preserved=True,
                  exhaustive=exhaustive_gates(ns), cpu_cpp=host_cpp_gate(ns, parts),
                  numerical=numerical_gates(tree, ns), wrappers=wrapper_gates(ns, parts), policy_cases=policy_gates(ns),
                  cuda_tools={**{name: shutil.which(name) for name in ('nvcc', 'ptxas')},
                              'nvrtc_library': ctypes.util.find_library('nvrtc')},
                  cuda_compile='SKIPPED: no CUDA/NVRTC toolchain',
                  gpu='NOT RUN: no GPU; no OKC calls authorized')
    (ROOT / 'results/dense-cg-rotcodes-offline.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
