"""L2P: prefetch the next projections' weights into L2 during latency-bound kernels (T = 1 / 8).

Parent: experiments/decode_moe/decoder_r2f.py (scored 2.0510x, sha256 75fa2a73...). Output decoder_l2p.py.

At 1 and 8 tokens attention and the attention-boundary norm barely use HBM. Right after the qkv projection, a
side-stream kernel issues cp.async.bulk.prefetch.L2 for the o-projection weights and (variants) the FP32 router
weight and a prefix of the pooler weights, so those reads hit L2 later. Prefetches change no values.

Prepare times a qkv -> 5 us window -> o -> 4.5 us window -> pooler+router chain (windows stand in for attention
and the norm) for each variant against no prefetch, checks every output bitwise, and enables the fastest variant
per token count only if it beats no prefetch by > 3%.
"""
import ast
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from harness import append_harness  # noqa: E402

PY_EDITS = [
    (" qkv = _achita_dense_project(data['attn_input'], 'qkv_gate', state)\n",
     " qkv = _achita_dense_project(data['attn_input'], 'qkv_gate', state)\n _l2p_side = _l2p_issue(tokens, state)\n"),
    (" return (next_attn_input, stream, topk_ids)\n",
     " _l2p_join(_l2p_side)\n return (next_attn_input, stream, topk_ids)\n"),
]

APPEND = r'''

# L2P (decode handoffs): side-stream L2 prefetch of the o / router / pooler weights while attention and the norm run.
import threading as _l2p_thr
import time as _l2p_time
_L2P_CUDA = r"""
#include <cuda_runtime.h>
#include <stdint.h>
struct L2pRegions { unsigned long long p[6]; unsigned long long b[6]; int n; };
__global__ void l2p_prefetch(L2pRegions r) {
  const unsigned long long CH = 32768ull;
  const unsigned long long g = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned long long G = (unsigned long long)gridDim.x * blockDim.x;
  for (int i = 0; i < r.n; ++i) {
    const unsigned long long nch = (r.b[i] + CH - 1) / CH;
    for (unsigned long long c = g; c < nch; c += G) {
      const unsigned long long off = c * CH;
      unsigned long long sz = r.b[i] - off < CH ? r.b[i] - off : CH;
      sz &= ~15ull;
      if (sz) asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(r.p[i] + off), "r"((unsigned)sz) : "memory");
    }
  }
}
__global__ void l2p_load(L2pRegions r) {
  // L2 warm-up with ordinary loads from every SM; the values are discarded (asm volatile keeps the loads)
  const unsigned long long g = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned long long G = (unsigned long long)gridDim.x * blockDim.x;
  unsigned acc = 0;
  for (int i = 0; i < r.n; ++i) {
    const uint4* p = reinterpret_cast<const uint4*>(r.p[i]);
    const unsigned long long n16 = r.b[i] / 16;
    unsigned long long j = g;
    for (; j + 3 * G < n16; j += 4 * G) {
      uint4 v[4];
#pragma unroll
      for (int u = 0; u < 4; ++u)
        asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(v[u].x), "=r"(v[u].y), "=r"(v[u].z), "=r"(v[u].w) : "l"(p + j + u * G));
#pragma unroll
      for (int u = 0; u < 4; ++u) acc ^= v[u].x;
    }
    for (; j < n16; j += G) {
      uint4 v;
      asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
                   : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p + j));
      acc ^= v.x;
    }
  }
  if (acc == 0x9e3779b9u && r.n < 0) r.p[0] = acc;
}
__global__ void l2p_spin(unsigned long long ns) {
  if (threadIdx.x) return;
  unsigned long long t0, t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
  do { asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); } while (t - t0 < ns);
}
int l2p_launch(int64_t p0, int64_t b0, int64_t p1, int64_t b1, int64_t p2, int64_t b2, int64_t p3, int64_t b3,
               int64_t p4, int64_t b4, int64_t p5, int64_t b5, int n, int blocks, int mode, int64_t stream) {
  L2pRegions r{};
  const int64_t ps[6] = {p0, p1, p2, p3, p4, p5}, bs[6] = {b0, b1, b2, b3, b4, b5};
  if (n < 0 || n > 6 || blocks < 1 || blocks > 132) return -1;
  for (int i = 0; i < 6; ++i) {
    if (i < n && ((ps[i] & 15) || bs[i] < 0)) return -2;
    r.p[i] = (unsigned long long)ps[i];
    r.b[i] = (unsigned long long)bs[i];
  }
  r.n = n;
  if (mode == 1) l2p_load<<<blocks, 256, 0, (cudaStream_t)stream>>>(r);
  else l2p_prefetch<<<blocks, 128, 0, (cudaStream_t)stream>>>(r);
  return cudaGetLastError() == cudaSuccess ? 0 : -8;
}
int l2p_spin_launch(int64_t ns, int64_t stream) {
  l2p_spin<<<1, 32, 0, (cudaStream_t)stream>>>((unsigned long long)ns);
  return cudaGetLastError() == cudaSuccess ? 0 : -8;
}
"""
_L2P_CPP = ('\n#include <pybind11/pybind11.h>\n#include <stdint.h>\nint l2p_launch(int64_t, int64_t, int64_t, int64_t, int64_t, '
            'int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int, int, int, int64_t);\nint l2p_spin_launch(int64_t, int64_t);\n')
_L2P_EXT = []


def _l2p_build():
    try:
        tag = hashlib.sha256(_L2P_CUDA.encode()).hexdigest()[:12]
        _L2P_EXT.append(load_inline(f'okc_l2p_{tag}', _L2P_CPP, cuda_sources=_L2P_CUDA, functions=['l2p_launch', 'l2p_spin_launch'],
                                    extra_cuda_cflags=['-O3'], no_implicit_headers=True, with_pytorch_error_handling=False,
                                    verbose=False))
    except Exception as e:
        print('L2P_BUILD_ERROR', repr(e)[-1800:], flush=True)


_l2p_thread = _l2p_thr.Thread(target=_l2p_build, daemon=True)
_l2p_thread.start()


def _l2p_go(plan, state):
    (mode, blocks, regions) = plan
    main, side = torch.cuda.current_stream(), state['side_stream']
    side.wait_stream(main)
    a = []
    for p, b in regions:
        a += [p, b]
    a += [0, 0] * (6 - len(regions))
    rc = _L2P_EXT[0].l2p_launch(*a, len(regions), blocks, mode, side.cuda_stream)
    if rc:
        raise RuntimeError(f'l2p launch failed: {rc}')
    return side


def _l2p_issue(tokens, state):
    plan = state.get('l2p', {}).get(tokens)
    return None if plan is None or not _L2P_EXT else _l2p_go(plan, state)


def _l2p_join(side):
    if side is not None:
        torch.cuda.current_stream().wait_stream(side)


def _l2p_regions(state, m, pool_mb, router):
    dense = state['achita_dense']
    if m == 1:
        op = (_r2_find(dense, 'v12_o1') or {}).get('plans', {}).get(('o', 1))
    else:
        op = ((_r2_find(dense, 'v18') or {}).get('plans') or {}).get(('o', 8))
    if op is None or m not in state.get('r2', {}):
        return None
    pp = state['r2'][m][1]
    nb = lambda t: t.numel() * t.element_size()
    regs = [(op['wp'].data_ptr(), nb(op['wp'])), (op['esc'].data_ptr(), nb(op['esc']))]
    if router:
        regs.append((state['rf']['weight'].data_ptr(), nb(state['rf']['weight'])))
    if pool_mb:
        regs += [(pp['esc'].data_ptr(), nb(pp['esc'])), (pp['wp'].data_ptr(), min(nb(pp['wp']), pool_mb << 20))]
    return regs


def _l2p_select(state):
    ext = _L2P_EXT[0]
    dev = state['rf']['weight'].device
    for m in (1, 8):
        if m not in state.get('r2sel', ()):
            print(f'L2P M={m} skip (R2 not selected)', flush=True)
            continue
        variants = {'none': None}
        for lab, mode, blocks, pmb, rt in (('b8_o', 0, 8, 0, False), ('b132_o', 0, 132, 0, False), ('b132_orp12', 0, 132, 12, True),
                                           ('ld_o', 1, 132, 0, False), ('ld_or', 1, 132, 0, True), ('ld_orp12', 1, 132, 12, True),
                                           ('ld_orp24', 1, 132, 24, True)):
            r = _l2p_regions(state, m, pmb, rt)
            if r is not None:
                variants[lab] = (mode, blocks, r)
        g = torch.Generator(device=dev).manual_seed(4100 + m)
        xq = torch.randn(m, 8192, device=dev, generator=g).to(torch.bfloat16)
        xo = torch.randn(m, 1024, device=dev, generator=g).to(torch.bfloat16)
        xb = torch.randn(m, 8192, device=dev, generator=g).to(torch.bfloat16)
        xr = torch.randn(m, 8192, device=dev, generator=g)

        def chain(regs):
            def f():
                q = _achita_dense_project(xq, 'qkv_gate', state)
                side = _l2p_go(regs, state) if regs is not None else None
                ext.l2p_spin_launch(5000, torch.cuda.current_stream().cuda_stream)
                o = _achita_dense_project(xo, 'o', state)
                ext.l2p_spin_launch(4500, torch.cuda.current_stream().cuda_stream)
                r = _r2_pool_route(xb, xr, state)
                _l2p_join(side)
                return None if r is None else (q, o) + tuple(r)
            return f

        ref = chain(None)()
        ref = None if ref is None else [t.clone() for t in ref]
        ok = {}
        for lab, regs in variants.items():
            got = chain(regs)()
            torch.cuda.synchronize()
            ok[lab] = ref is not None and _okh_same(ref, got)
        med = _okh_time({lab: chain(regs) for lab, regs in variants.items()}, dev, reps=30)
        pick = _okh_pick(med, 'none', 0.97, allowed=[k for k in variants if k != 'none' and ok[k]])
        mb = {lab: sum(b for _, b in regs[2]) / 2 ** 20 for lab, regs in variants.items() if regs is not None}
        print(f'L2P M={m} ' + ' '.join(f'{k}={v:.2f}us' + ('' if ok[k] else '(differs)') for k, v in med.items())
              + ' MB ' + ' '.join(f'{k}={v:.1f}' for k, v in mb.items()) + f' pick={pick}', flush=True)
        if pick != 'none':
            state['l2p'][m] = variants[pick]


_l2p_prev_prepare = prepare


def prepare(weights):
    t0 = _l2p_time.perf_counter()
    state = _l2p_prev_prepare(weights)
    state['l2p'] = {}
    _l2p_thread.join(max(0.0, 280.0 - (_l2p_time.perf_counter() - t0)))
    el = _l2p_time.perf_counter() - t0
    if not _L2P_EXT or el > 278.0 or state.get('rf') is None:
        print(f'L2P_SKIP {el:.1f}s built={bool(_L2P_EXT)}', flush=True)
        return state
    try:
        _l2p_select(state)
    except Exception as e:
        state['l2p'] = {}
        print('L2P_ERROR', repr(e)[:400], flush=True)
    torch.cuda.synchronize()
    print(f"L2P_DONE {_l2p_time.perf_counter() - t0:.1f}s sel={sorted(state['l2p'])}", flush=True)
    return state
'''


def compact(app):
    lines = []
    for ln in app.split('\n'):
        n = len(ln) - len(ln.lstrip(' '))
        lines.append(' ' * (n // 4) + ln.lstrip(' ') if n % 4 == 0 else ln)
    return '\n'.join(lines)


def main():
    src = (HERE / 'decoder_r2f.py').read_bytes().decode('utf-8')
    assert hashlib.sha256(src.encode()).hexdigest().startswith('75fa2a73'), 'must start from the scored R2f file'
    out = src
    for old, new in PY_EDITS:
        assert out.count(old) == 1, old
        out = out.replace(old, new)
    out = append_harness(out) + compact(APPEND)
    ast.parse(out)
    b = out.encode('utf-8')
    (HERE / 'decoder_l2p.py').write_bytes(b)
    cu = APPEND[APPEND.index('_L2P_CUDA = r"""') + 16:APPEND.index('"""\n_L2P_CPP')]
    (HERE / 'l2p.cu').write_text(cu, encoding='utf-8', newline='')
    print(json.dumps({'sha256': hashlib.sha256(b).hexdigest(), 'bytes': len(b), 'headroom': 1_000_000 - len(b)}))


if __name__ == '__main__':
    main()
