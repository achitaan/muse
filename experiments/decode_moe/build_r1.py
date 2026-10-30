"""Build R1: small-token router/pooler co-scheduling candidate for muse_decode_layer.

Parent: board_best/decoder_finreg_fp8pv_scored_2.0071.py (scored 2.0071x, job 646774bad23e4c7f).

At T=1 and T=8 the exact FP32 router (`route_fp32`, 128 CTAs on the side stream) and the pooler GEMV
(main stream) contend for SMs; the saved profiles show the router spanning 10-17 us for 8.4 MB of weights.
R1 changes only scheduling, never arithmetic:
  * `route_fp32<TM, BPC>`: each CTA processes BPC of the original 128 (row-tile, K-split) blocks in sequence
    (grid 128/BPC). Every partial uses the identical FMA/shuffle order and the last-CTA reduction is
    unchanged, so router outputs are bitwise identical for every BPC.
  * order 1 enqueues the router before the pooler (so its CTAs are resident first); order 0 is the parent.
A prepare-time harness checks every variant bitwise against the parent order on random inputs, times all of
them as interleaved cold-L2 CUDA-graph replays, prints the table, and selects a variant per token count only
if it beats the parent arrangement by more than 3%; otherwise the parent arrangement is kept.
"""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PARENT = ROOT / 'board_best/decoder_finreg_fp8pv_scored_2.0071.py'
OUT = HERE / 'decoder_r1.py'


def const_seg(src, name):
    t = ast.parse(src)
    n = next(n for n in t.body if isinstance(n, ast.Assign) and getattr(n.targets[0], 'id', '') == name)
    seg = ast.get_source_segment(src, n.value)
    assert src.count(seg) == 1
    return seg, n.value.value


def sub(src, old, new):
    assert src.count(old) == 1, old[:80]
    return src.replace(old, new)


CUDA_EDITS = [
    ("""template<int TM>
__global__ void route_fp32(const float* __restrict__ x, const float* __restrict__ w,
                          const float* __restrict__ bias, volatile float* parts,
                          unsigned int* counter, float* outw, int* outi, int M) {
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int nb = blockIdx.x & 15, ks = blockIdx.x >> 4;
""", """template<int TM, int BPC>
__global__ void route_fp32(const float* __restrict__ x, const float* __restrict__ w,
                          const float* __restrict__ bias, volatile float* parts,
                          unsigned int* counter, float* outw, int* outi, int M) {
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  // R1: BPC consecutive (row tile, K split) blocks per CTA, each with the original arithmetic.
#pragma unroll 1
  for (int bi = 0; bi < BPC; ++bi) {
  const int blk = blockIdx.x * BPC + bi;
  const int nb = blk & 15, ks = blk >> 4;
"""),
    ("""        parts[((size_t)ks * 8 + m) * 256 + row1] = b[m];
      }
    }
  }
""", """        parts[((size_t)ks * 8 + m) * 256 + row1] = b[m];
      }
    }
  }
  }
"""),
    ("""    unsigned int old = atomicAdd(counter, 1u);
    last = ((old & 127u) == 127u);""", """    unsigned int old = atomicAdd(counter, (unsigned int)BPC);
    last = (((old + (unsigned int)BPC) & 127u) == 0u);"""),
    ("""         at::Tensor counter, at::Tensor outw, at::Tensor outi) {""",
     """         at::Tensor counter, at::Tensor outw, at::Tensor outi, int64_t bpc) {"""),
    ("""  if (M == 1) route_fp32<1><<<128, 256, 0, stream>>>(x.data_ptr<float>(), w.data_ptr<float>(), bias.data_ptr<float>(),
      p, c, outw.data_ptr<float>(), outi.data_ptr<int>(), M);
  else route_fp32<8><<<128, 256, 0, stream>>>(x.data_ptr<float>(), w.data_ptr<float>(), bias.data_ptr<float>(),
      p, c, outw.data_ptr<float>(), outi.data_ptr<int>(), M);""",
     """#define RF_L(T, B) route_fp32<T, B><<<128 / B, 256, 0, stream>>>(x.data_ptr<float>(), w.data_ptr<float>(), \\
      bias.data_ptr<float>(), p, c, outw.data_ptr<float>(), outi.data_ptr<int>(), M)
#define RF_S(T) switch (bpc) { case 2: RF_L(T, 2); break; case 4: RF_L(T, 4); break; case 8: RF_L(T, 8); break; \\
      default: RF_L(T, 1); }
  if (M == 1) { RF_S(1) } else { RF_S(8) }"""),
]

PY_EDITS = [
    ("""  side_stream.wait_stream(main_stream)
  expert_input = _achita_dense_project(ffn_input, 'pooler', state)
  with torch.cuda.stream(side_stream):
   if not split_router and tokens > 0 and (state['rf'] is not None):
    (topk_weights, topk_ids) = _rf_route(router_input, state['rf'])
   else:""", """  side_stream.wait_stream(main_stream)
  (r1o, r1b) = state.get('r1', {}).get(tokens, (0, 1)) if not split_router and tokens > 0 and (state['rf'] is not None) else (0, 1)
  if r1o == 2:
   (topk_weights, topk_ids) = _rf_route(router_input, state['rf'], r1b)
  expert_input = _achita_dense_project(ffn_input, 'pooler', state)
  if r1o == 3:
   (topk_weights, topk_ids) = _rf_route(router_input, state['rf'], r1b)
  with torch.cuda.stream(side_stream):
   if r1o >= 2:
    pass
   elif not split_router and tokens > 0 and (state['rf'] is not None):
    (topk_weights, topk_ids) = _rf_route(router_input, state['rf'], r1b)
   else:"""),
    ("""def _rf_route(x, state):""", """def _rf_route(x, state, bpc=1):"""),
    ("""state['counter'], outw, outi)
 return (outw, outi)""", """state['counter'], outw, outi, bpc)
 return (outw, outi)"""),
    ("""torch::Tensor counter, torch::Tensor outw, torch::Tensor outi);""",
     """torch::Tensor counter, torch::Tensor outw, torch::Tensor outi, int64_t bpc);"""),
]

HARNESS = r'''

# R1 (decode-moe-plan): prepare-time selection of the small-token router/pooler arrangement.
import time as _r1_time
_R1_VARIANTS = ((0, 1), (1, 1), (2, 1), (3, 1), (2, 2), (0, 4))
_R1_SAFE = ((2, 1), (3, 1), (2, 2), (0, 4))  # o1 (router first on the side stream) failed a judge recheck: never selected


def _r1_pair(xb, xr, state, o, b):
 main, side = torch.cuda.current_stream(), state['side_stream']
 side.wait_stream(main)
 if o == 1:
  with torch.cuda.stream(side):
   r = _rf_route(xr, state['rf'], b)
 if o == 2:
  r = _rf_route(xr, state['rf'], b)
 e = _achita_dense_project(xb, 'pooler', state)
 if o == 3:
  r = _rf_route(xr, state['rf'], b)
 if o == 0:
  with torch.cuda.stream(side):
   r = _rf_route(xr, state['rf'], b)
 main.wait_stream(side)
 return (e, r[0], r[1])


def _r1_select(state, reps=25):
 sel = {}
 dev = state['rf']['weight'].device
 flush = torch.empty(256 << 20, dtype=torch.uint8, device=dev)
 for M in (1, 8):
  g = torch.Generator(device=dev).manual_seed(1000 + M)
  xb = torch.randn(M, 8192, device=dev, generator=g).to(torch.bfloat16)
  xr = torch.randn(M, 8192, device=dev, generator=g)
  ref = [t.clone() for t in _r1_pair(xb, xr, state, 0, 1)]
  graphs = {}
  for v in _R1_VARIANTS:
   out = _r1_pair(xb, xr, state, *v)
   torch.cuda.synchronize()
   if not all(torch.equal(a, b) for a, b in zip(ref, out)):
    print(f'R1_MISMATCH M={M} v={v}', flush=True)
    continue
   gr = torch.cuda.CUDAGraph()
   with torch.cuda.graph(gr):
    _r1_pair(xb, xr, state, *v)
   graphs[v] = gr
  evs = []
  for _ in range(reps):
   for v, gr in graphs.items():
    flush.zero_()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record(); gr.replay(); b.record()
    evs.append((v, a, b))
  torch.cuda.synchronize()
  times = {v: [] for v in graphs}
  for v, a, b in evs:
   times[v].append(a.elapsed_time(b) * 1000.0)
  med = {v: sorted(t)[len(t) // 2] for v, t in times.items()}
  print(f'R1_TIMES M={M} ' + ' '.join(f'o{v[0]}b{v[1]}={t:.2f}' for v, t in med.items()), flush=True)
  base = med.get((0, 1))
  safe = {v: t for v, t in med.items() if v in _R1_SAFE}
  best = min(safe, key=safe.get) if safe else (0, 1)
  if base is not None and best != (0, 1) and safe and med[best] < 0.97 * base:
   sel[M] = best
  print(f'R1_SELECT M={M} {sel.get(M, (0, 1))}', flush=True)
  del graphs
 return sel


_r1_prev_prepare = prepare


def prepare(weights):
 t0 = _r1_time.perf_counter()
 state = _r1_prev_prepare(weights)
 state['r1'] = {}
 el = _r1_time.perf_counter() - t0
 if state.get('rf') is None or el > 280.0:
  print(f'R1_SKIP {el:.1f}s', flush=True)
  return state
 try:
  state['r1'] = _r1_select(state)
 except Exception as e:
  print('R1_ERROR', repr(e), flush=True)
  state['r1'] = {}
 torch.cuda.synchronize()
 print(f'R1_DONE {_r1_time.perf_counter() - t0:.1f}s {state["r1"]}', flush=True)
 return state
'''


def main():
    # Read the committed blob (exact scored bytes, LF) rather than the possibly CRLF-converted checkout.
    import subprocess
    raw = subprocess.run(['git', 'cat-file', 'blob', 'f4c1921db2915294075737656e4a5201b3147598'], cwd=ROOT,
                         check=True, capture_output=True).stdout
    assert hashlib.sha256(raw).hexdigest().startswith('648f8cee')
    src = raw.decode('utf-8')
    seg, cuda = const_seg(src, '_RF_CUDA')
    new_cuda = cuda
    for old, new in CUDA_EDITS:
        new_cuda = sub(new_cuda, old, new)
    out = src.replace(seg, repr(new_cuda))
    for old, new in PY_EDITS:
        out = sub(out, old, new)
    name_old = "okc_router_fp32_d5a88ed70badd431_90a"
    tag = hashlib.sha256(new_cuda.encode()).hexdigest()[:16]
    out = sub(out, name_old, f"okc_router_fp32_r1{tag}_90a")
    out = out.rstrip('\n') + '\n' + HARNESS
    ast.parse(out)
    (HERE / 'rf_r1.cu').write_text(new_cuda, encoding='utf-8', newline='')
    OUT.write_bytes(out.encode('utf-8'))
    b = out.encode('utf-8')
    print(json.dumps({'parent_sha256': hashlib.sha256(src.encode()).hexdigest(), 'out': str(OUT.name),
                      'sha256': hashlib.sha256(b).hexdigest(), 'bytes': len(b), 'limit_headroom': 1_000_000 - len(b)},
                     indent=1))


if __name__ == '__main__':
    main()
