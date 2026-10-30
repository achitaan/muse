"""Build R2: exact FP32 router fused into the small-token pooler GEMV (T = 1 and T = 8) for muse_decode_layer.

Parent: board_best/decoder_finreg_fp8pv_scored_2.0071.py (blob f4c1921d, scored 2.0071x).

The parent runs the pooler (v18s_gemv at T = 1, smallm2_gemv at T = 8: 128 CTAs x 1024 threads, which
fill the GPU's register files) and the exact FP32 router (`route_fp32`, 128 CTAs) as two kernels on
two streams. The router CTAs find no room and wait for the pooler, so they run almost serially.
Launching the router first saved 6-7 us, but it failed a judge recheck (job 209ff320073b48f6).

R2 runs both in ONE launch. Its pooler grid is exactly the router's 128 (row tile, K split)
blocks:
  * warps 0..15 of CTA b first compute router rows nb*16 + warp of K-slice ks = b >> 4, with the
    identical per-row FMA order, warp-shuffle reduction and parts layout as route_fp32;
  * a named barrier (bar.sync 1, 512: only those 512 threads) publishes them, and one atomic per CTA
    counts arrivals. The last CTA's warps 0..M-1 then run route_fp32's exact top-8 tail;
  * all 32 warps then do the unchanged GEMV (a separate template instance, OPT bit 64).
The modified CUDA is derived at import from the parent's own templates by exact-anchor replacements.
It is built on a background thread, so it does not lengthen the serial prepare path.

Prepare-time gate: pooler output, routing weights and ids must be bitwise equal to the parent pair
(pooler + route_fp32) on random inputs at T = 1 and 8. It is used only if interleaved cold-L2 graph
timing beats the parent pair by more than 3%. Any failure keeps the parent path.
"""
import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

PY_EDITS = [
    (""" if 8 < tokens <= 32 and state['fpr'] is not None:
  (expert_input, logits) = _fpr_bfp_fused_pooler_router(ffn_input, state['fpr'])
  (topk_weights, topk_ids) = state['routing'](None, logits, TOP_K, True)
 else:
""", """ if 8 < tokens <= 32 and state['fpr'] is not None:
  (expert_input, logits) = _fpr_bfp_fused_pooler_router(ffn_input, state['fpr'])
  (topk_weights, topk_ids) = state['routing'](None, logits, TOP_K, True)
 elif not split_router and tokens in state.get('r2sel', ()) and (r2 := _r2_pool_route(ffn_input, router_input, state)) is not None:
  (expert_input, topk_weights, topk_ids) = r2
 else:
"""),
]

APPEND = r'''

# R2 (decode-moe-plan Phase 3): exact FP32 router fused into the T = 1 / 8 pooler GEMV; gated in prepare.
import threading as _r2_thr
import time as _r2_time
_R2_DEV = r"""
struct R2Args { const float* x; const float* w; const float* bias; volatile float* parts; unsigned int* counter;
                float* outw; int* outi; int M; int dbg; };
__device__ __forceinline__ void P_r2_router(const R2Args& ra, int warp, int lane) {
  // R2b: the CTA's x slice is staged in shared memory once and each router warp issues its 32 weight loads
  // up front; the per-row FMA order (j = 0..31, then the shuffle tree) is route_fp32's, so results are bitwise equal
  __shared__ float r2x[8 * 1024];
  const int blk = blockIdx.y * gridDim.x + blockIdx.x, nb = blk & 15, ks = blk >> 4, row = nb * 16 + warp;
  const int M = ra.M;
  for (int i = threadIdx.x; i < M * 1024; i += blockDim.x) r2x[i] = ra.x[(size_t)(i >> 10) * 8192 + ks * 1024 + (i & 1023)];
  __syncthreads();
  if (warp >= 16) return;
  float a[8];
#pragma unroll
  for (int m = 0; m < 8; ++m) a[m] = 0.f;
  if (!(ra.dbg & 2)) {
  float wv[32];
#pragma unroll
  for (int j = 0; j < 32; ++j) wv[j] = __ldg(ra.w + (size_t)row * 8192 + ks * 1024 + j * 32 + lane);
#pragma unroll
  for (int j = 0; j < 32; ++j) {
#pragma unroll
    for (int m = 0; m < 8; ++m)
      if (m < M) a[m] = __fmaf_rn(r2x[m * 1024 + j * 32 + lane], wv[j], a[m]);
  }
  }
#pragma unroll
  for (int m = 0; m < 8; ++m) {
    if (m < M) {
#pragma unroll
      for (int d = 16; d > 0; d >>= 1) a[m] += __shfl_down_sync(0xffffffffu, a[m], d);
      if (lane == 0) ra.parts[((size_t)ks * 8 + m) * 256 + row] = a[m];
    }
  }
  if (lane == 0) __threadfence();
  asm volatile("bar.sync 1, 512;" ::: "memory");
  __shared__ int r2_last;
  if (warp == 0 && lane == 0) r2_last = ((atomicAdd(ra.counter, 1u) & 127u) == 127u);
  asm volatile("bar.sync 1, 512;" ::: "memory");
  if (!r2_last || warp >= M || (ra.dbg & 1)) return;
  __threadfence();
  float score[8], sel[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const int e = lane + 32 * j;
    float v = 0.f;
#pragma unroll
    for (int s = 0; s < 8; ++s) v += __ldcg((const float*)ra.parts + ((size_t)s * 8 + warp) * 256 + e);  // R2e: L2 loads, pipelined (order unchanged)
    const float z = __fmul_rn(v, 0.1767766952966369f);
    score[j] = 1.f / (1.f + expf(-z));
    sel[j] = score[j] + ra.bias[e];
  }
  float chosen_score[8];
  int chosen_id[8];
  float sum = 0.f;
#pragma unroll
  for (int r = 0; r < 8; ++r) {
    float best = __int_as_float(0xff800000), raw = 0.f;
    int id = 2147483647;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int e = lane + j * 32;
      if (sel[j] > best || (sel[j] == best && e < id)) { best = sel[j]; raw = score[j]; id = e; }
    }
#pragma unroll
    for (int d = 16; d > 0; d >>= 1) {
      const float ob = __shfl_xor_sync(0xffffffffu, best, d);
      const float os = __shfl_xor_sync(0xffffffffu, raw, d);
      const int oi = __shfl_xor_sync(0xffffffffu, id, d);
      if (ob > best || (ob == best && oi < id)) { best = ob; raw = os; id = oi; }
    }
    chosen_score[r] = raw; chosen_id[r] = id; sum += raw;
#pragma unroll
    for (int j = 0; j < 8; ++j) if (lane + j * 32 == id) sel[j] = __int_as_float(0xff800000);
  }
  const float inv = 1.f / (sum + 1e-15f);
#pragma unroll
  for (int r = 0; r < 8; ++r)
    if (lane == r) { ra.outw[warp * 8 + r] = chosen_score[r] * inv; ra.outi[warp * 8 + r] = chosen_id[r]; }
}
"""
_R2_HOST = r"""
int r2_launch(int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU, int S,
              int64_t t01, int64_t u01, int64_t stream, int64_t rx, int64_t rw, int64_t rb, int64_t rp, int64_t rc,
              int64_t ow, int64_t oi, int dbg) {
  if ((N / (WR * 16)) * S != 128 || WR * WK * 32 < 512 || m < 1 || m > 8) return -20;
  R2Args ra{(const float*)rx, (const float*)rw, (const float*)rb, (volatile float*)rp, (unsigned int*)rc,
            (float*)ow, (int*)oi, m, dbg};
  return P_launch_t<@@ARGS@@>(wp, x, out, esc, m, N, K, WR, WK, KU, S, t01, u01, stream, 0, ra);
}
"""
_R2_CPP = ('\n#include <pybind11/pybind11.h>\n#include <stdint.h>\nint r2_launch(int64_t, int64_t, int64_t, int64_t, int, int, int, int, int, int, int, '
           'int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int);\n')


def _r2_source(tmpl, p, inst):
    if inst[1] != 1 or inst[2] != 1:  # router mapping assumes MT = RT = 1 (128 CTAs = N / (WR * 16) * S)
        return None
    s = tmpl.replace('@@CASES@@', '')
    kh = 'template <int FMT, int MT, int RT, int ST, int RED, int DIAG, int NTMAX, int CM, int CN, int CK, int CWR, int CWK, int CKU, int CS, int OPT>\n__global__'
    edits = [(kh, _R2_DEV.replace('P_', p + '_') + kh),
             ('uint32_t T0, uint32_t T1,\n    uint32_t U0, uint32_t U1) {\n', 'uint32_t T0, uint32_t T1,\n    uint32_t U0, uint32_t U1, R2Args ra) {\n'),
             ('lane = tid & 31, nt = CWR ? CWR * CWK * 32 : blockDim.x;\n',
              'lane = tid & 31, nt = CWR ? CWR * CWK * 32 : blockDim.x;\n    if constexpr ((OPT & 64) != 0) ' + p + '_r2_router(ra, warp, lane);\n'),
             ('(uint32_t)((uint64_t)u01 >> 32));', '(uint32_t)((uint64_t)u01 >> 32), ra);')]
    for a, b in edits:
        if s.count(a) != 1:
            return None
        s = s.replace(a, b)
    a = 'int S, int64_t t01, int64_t u01, int64_t stream, int query) {'
    i = s.find(a)
    if i < 0 or s.find('static int ' + p + '_launch_t(', 0, i) < 0:
        return None
    s = s[:i] + 'int S, int64_t t01, int64_t u01, int64_t stream, int query, R2Args ra) {' + s[i + len(a):]
    s = s.replace('switch (inst) {', 'R2Args ra{}; (void)ra;\n  switch (inst) {', 1)
    args = ', '.join(str(v) for v in inst[:14] + (inst[14] | 64,))
    return s + _R2_HOST.replace('P_', p + '_').replace('@@ARGS@@', args)


def _r2_find(d, key, depth=0):
    if not isinstance(d, dict) or depth > 6:
        return None
    if key in d:
        return d[key]
    for v in d.values():
        r = _r2_find(v, key, depth + 1)
        if r is not None:
            return r
    return None


_R2_EXT = {}


def _r2_build():
    for m, tmpl, p, inst in ((1, V18S_CUDA_TEMPLATE, 'v18s', V18_SMALL[('pooler', 1)][0]),
                             (8, tt_v10_sm2_SMALLM2_CUDA_TEMPLATE, 'smallm2', tt_v10_sm2_SMALLM2_ROUTES[('pooler', 8)][0])):
        try:
            src = _r2_source(tmpl, p, inst)
            if src is not None:
                tag = hashlib.sha256(src.encode()).hexdigest()[:12]
                _R2_EXT[m] = (inst, load_inline(f'okc_r2_{p}_{tag}', _R2_CPP, cuda_sources=src, functions=['r2_launch'],
                              extra_cuda_cflags=['-O3', '-lineinfo'], no_implicit_headers=True,
                              with_pytorch_error_handling=False, verbose=False))
        except Exception as e:
            print('R2_BUILD_ERROR', m, repr(e)[-1800:], flush=True)


import hashlib
_r2_thread = _r2_thr.Thread(target=_r2_build, daemon=True)
_r2_thread.start()


def _r2_pool_route(xb, xr, state):
    e = state.get('r2', {}).get(xb.shape[0])
    if e is None or xb.dtype != torch.bfloat16 or not xb.is_contiguous() or xb.data_ptr() % 16 or xb.shape[1] != 8192:
        return None
    if xr is None or xr.dtype != torch.float32 or tuple(xr.shape) != (xb.shape[0], 8192) or not xr.is_contiguous():
        return None
    (ext, plan) = e
    rf = state['rf']
    (m, dev) = (xb.shape[0], xb.device)
    out = torch.empty((m, 4096), dtype=torch.bfloat16, device=dev)
    (ow, oi) = (torch.empty((m, 8), dtype=torch.float32, device=dev), torch.empty((m, 8), dtype=torch.int32, device=dev))
    (wr, wk, ku, s) = plan['cfg']
    rc = ext.r2_launch(plan['wp'].data_ptr(), xb.data_ptr(), out.data_ptr(), plan['esc'].data_ptr(), m, 4096, 8192, wr, wk,
                       ku, s, plan['t01'], plan['u01'], torch.cuda.current_stream().cuda_stream, xr.data_ptr(),
                       rf['weight'].data_ptr(), rf['bias'].data_ptr(), rf['parts'].data_ptr(), rf['counter'].data_ptr(),
                       ow.data_ptr(), oi.data_ptr(), state.get('r2dbg', 0))
    return None if rc else (out, ow, oi)


def _r2_parent(xb, xr, state):
    main, side = torch.cuda.current_stream(), state['side_stream']
    side.wait_stream(main)
    e = _achita_dense_project(xb, 'pooler', state)
    with torch.cuda.stream(side):
        r = _rf_route(xr, state['rf'])
    main.wait_stream(side)
    return (e, r[0], r[1])


def _r2_gate(state):
    dense = state['achita_dense']
    plans = {1: (_r2_find(dense, 'v18') or {}).get('plans', {}).get(('pooler', 1)),
             8: (_r2_find(dense, 'v10_sm2') or {}).get('plans', {}).get(('pooler', 8))}
    state['r2'] = {}
    for m, (inst, ext) in _R2_EXT.items():
        plan = plans.get(m)
        if plan is not None and plan.get('fmt') == inst[0] and tuple(plan['cfg']) == tuple(inst[10:14]):
            state['r2'][m] = (ext, plan)
    dev = state['rf']['weight'].device
    flush = torch.empty(256 << 20, dtype=torch.uint8, device=dev)
    for m in sorted(state['r2']):
        ok = True
        for seed in range(3):
            g = torch.Generator(device=dev).manual_seed(900 + 10 * m + seed)
            xb = torch.randn(m, 8192, device=dev, generator=g).to(torch.bfloat16)
            xr = torch.randn(m, 8192, device=dev, generator=g)
            ref = [t.clone() for t in _r2_parent(xb, xr, state)]
            got = _r2_pool_route(xb, xr, state)
            torch.cuda.synchronize()
            ok = ok and got is not None and all(torch.equal(a, b) for a, b in zip(ref, got))
        if not ok:
            print(f'R2_GATE M={m} FAIL', flush=True)
            continue
        graphs = {}
        pool = lambda b, r, s: _achita_dense_project(b, 'pooler', s)
        for lab, fn, dbg in (('parent', _r2_parent, 0), ('fused', _r2_pool_route, 0), ('pool_only', pool, 0),
                             ('f_notopk', _r2_pool_route, 1), ('f_nomath', _r2_pool_route, 2), ('f_neither', _r2_pool_route, 3)):
            state['r2dbg'] = dbg  # timing-only diagnostics: captured into their own graphs, never used by kernel_fn
            fn(xb, xr, state)
            torch.cuda.synchronize()
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr):
                fn(xb, xr, state)
            graphs[lab] = gr
        state['r2dbg'] = 0
        ts = {lab: [] for lab in graphs}
        evs = []
        for _ in range(25):
            for lab, gr in graphs.items():
                flush.zero_()
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record(); gr.replay(); b.record()
                evs.append((lab, a, b))
        torch.cuda.synchronize()
        for lab, a, b in evs:
            ts[lab].append(a.elapsed_time(b) * 1000.0)
        med = {lab: sorted(v)[len(v) // 2] for lab, v in ts.items()}
        use = med['fused'] < 0.97 * med['parent']
        if use:
            state['r2sel'] = state.get('r2sel', ()) + (m,)
        print(f"R2_GATE M={m} PASS parent={med['parent']:.2f}us fused={med['fused']:.2f}us use={use}", flush=True)
        print(f'R2_DIAG M={m} ' + ' '.join(f'{k}={v:.2f}us' for k, v in med.items()), flush=True)
        del graphs


_r2_prev_prepare = prepare


def prepare(weights):
    t0 = _r2_time.perf_counter()
    state = _r2_prev_prepare(weights)
    (state['r2'], state['r2sel']) = ({}, ())
    _r2_thread.join(max(0.0, 282.0 - (_r2_time.perf_counter() - t0)))
    el = _r2_time.perf_counter() - t0
    if _r2_thread.is_alive() or state.get('rf') is None or el > 283.0:
        print(f'R2_SKIP {el:.1f}s built={sorted(_R2_EXT)}', flush=True)
        return state
    try:
        _r2_gate(state)
    except Exception as e:
        (state['r2'], state['r2sel']) = ({}, ())
        print('R2_ERROR', repr(e)[:300], flush=True)
    torch.cuda.synchronize()
    print(f"R2_DONE {_r2_time.perf_counter() - t0:.1f}s sel={state['r2sel']}", flush=True)
    return state
'''


def main():
    raw = subprocess.run(['git', 'cat-file', 'blob', 'f4c1921db2915294075737656e4a5201b3147598'], cwd=ROOT,
                         check=True, capture_output=True).stdout
    assert hashlib.sha256(raw).hexdigest().startswith('648f8cee')
    out = raw.decode('utf-8')
    for old, new in PY_EDITS:
        assert out.count(old) == 1
        out = out.replace(old, new)
    # size: store the MoE CUDA string zlib-compressed (identical bytes after decompression; the parent already
    # stores its Dense CUDA templates this way)
    import base64 as _b, zlib as _z
    tt = ast.parse(out)
    nd = next(n for n in tt.body if isinstance(n, ast.Assign) and getattr(n.targets[0], 'id', '') == '_native_moe_CUDA_SRC')
    seg = ast.get_source_segment(out, nd.value)
    assert out.count(seg) == 1
    enc = _b.b64encode(_z.compress(nd.value.value.encode(), 9)).decode()
    out = out.replace(seg, "__import__('zlib').decompress(__import__('base64').b64decode('" + enc + "')).decode()")
    app = APPEND
    import base64 as _b64, zlib as _zl
    for name in ('_R2_DEV', '_R2_HOST'):
        a = app.index(name + ' = r"""')
        b = app.index('"""', a + len(name) + 7) + 3
        body = app[a + len(name) + 7:b - 3]
        enc = _b64.b64encode(_zl.compress(body.encode(), 9)).decode()
        app = app[:a] + name + ' = _zsrc(' + repr(enc) + ')' + app[b:]
    lines = []
    for ln in app.split('\n'):
        n = len(ln) - len(ln.lstrip(' '))
        lines.append(' ' * (n // 4) + ln.lstrip(' ') if n % 4 == 0 else ln)
    app = '\n'.join(lines)
    out = out.rstrip('\n') + '\n' + app
    ast.parse(out)
    b = out.encode('utf-8')
    (HERE / 'decoder_r2e.py').write_bytes(b)
    # emit the derived CUDA for offline NVRTC checks
    ns = {}
    t = ast.parse(out)
    src_mod = {'V18S_CUDA_TEMPLATE': None, 'tt_v10_sm2_SMALLM2_CUDA_TEMPLATE': None}
    import base64
    import zlib
    for n in t.body:
        if isinstance(n, ast.Assign) and getattr(n.targets[0], 'id', '') in src_mod:
            src_mod[n.targets[0].id] = zlib.decompress(base64.b64decode(n.value.args[0].value)).decode()
    fn_src = APPEND[APPEND.index('_R2_DEV = r"""'):APPEND.index('def _r2_find')]
    exec(compile(fn_src, 'r2', 'exec'), ns)
    for p, key, inst in (('v18s', 'V18S_CUDA_TEMPLATE', (1, 1, 1, 1, 0, 0, 1024, 0, 4096, 8192, 2, 16, 8, 1, 6)),
                         ('smallm2', 'tt_v10_sm2_SMALLM2_CUDA_TEMPLATE', (1, 1, 1, 1, 0, 0, 1024, 0, 4096, 8192, 2, 16, 8, 1, 6))):
        s = ns['_r2_source'](src_mod[key], p, inst)
        assert s is not None, p
        (HERE / f'r2e_{p}.cu').write_text(s, encoding='utf-8', newline='')
    print(json.dumps({'sha256': hashlib.sha256(b).hexdigest(), 'bytes': len(b), 'headroom': 1_000_000 - len(b)}))


if __name__ == '__main__':
    main()
