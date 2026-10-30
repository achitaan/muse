"""Build S1: prepare-time A/B of the small-token `_sfn_kernel` (Phase 3 `_sfn` item), on top of scored R2.

Parent: experiments/decode_moe/decoder_r2.py (scored 2.0339x, job 2027d3dda5664795).
`_sfn_kernel` (post-FFN norm + residual + next pre-attention norm at tokens <= 8) runs one 8-warp Triton
program per token: 1 or 8 CTAs on a 132-SM GPU, 3.3-3.9 us in every R2 profile. The prepare harness times
num_warps 4 / 8 / 16 / 32 as interleaved cold-L2 CUDA graphs at T = 1 and 8, and switches only to a variant
whose two outputs are bitwise equal to the 8-warp parent on random inputs and that is > 3% faster at both
token counts. Otherwise the parent setting stays; the printed table is the measured closure.
"""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

EDIT = ("""        cached['post_eps'], cached['pre_eps'], num_warps=8, enable_fp_fusion=False)""",
        """        cached['post_eps'], cached['pre_eps'], num_warps=state.get('sfn_nw', 8), enable_fp_fusion=False)""")

APPEND = r'''

# S1 (decode-moe-plan Phase 3): prepare-time A/B of _sfn_kernel warps; bitwise + > 3% gated.
def _s1_select(state):
 if state.get('sliced_ffn_boundary') is None:
  return
 dev = state['rf']['weight'].device if state.get('rf') else torch.device('cuda')
 flush = torch.empty(256 << 20, dtype=torch.uint8, device=dev)
 med_all, ok_all = {}, {}
 for T in (1, 8):
  g = torch.Generator(device=dev).manual_seed(77 + T)
  br = torch.randn(T, UNPOOL_SLICE, device=dev, generator=g).to(torch.bfloat16)
  st = torch.randn(T, HIDDEN, device=dev, generator=g)
  outs, graphs = {}, {}
  for nw in (8, 4, 16, 32):
   state['sfn_nw'] = nw
   o = [t.clone() for t in _sfn_boundary(st, br, state)]
   torch.cuda.synchronize()
   outs[nw] = o
   gr = torch.cuda.CUDAGraph()
   with torch.cuda.graph(gr):
    _sfn_boundary(st, br, state)
   graphs[nw] = gr
  state['sfn_nw'] = 8
  evs = []
  for _ in range(30):
   for nw, gr in graphs.items():
    flush.zero_()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record(); gr.replay(); b.record()
    evs.append((nw, a, b))
  torch.cuda.synchronize()
  ts = {nw: [] for nw in graphs}
  for nw, a, b in evs:
   ts[nw].append(a.elapsed_time(b) * 1000.0)
  med = {nw: sorted(v)[len(v) // 2] for nw, v in ts.items()}
  ok = {nw: all(torch.equal(x, y) for x, y in zip(outs[8], outs[nw])) for nw in outs}
  med_all[T], ok_all[T] = med, ok
  print('S1_SFN T=%d ' % T + ' '.join('nw%d=%.2fus%s' % (nw, med[nw], '' if ok[nw] else '(differs)') for nw in med), flush=True)
  del graphs
 best = None
 for nw in (4, 16, 32):
  if all(ok_all[T][nw] and med_all[T][nw] < 0.97 * med_all[T][8] for T in (1, 8)):
   if best is None or sum(med_all[T][nw] for T in (1, 8)) < sum(med_all[T][best] for T in (1, 8)):
    best = nw
 state['sfn_nw'] = best or 8
 print('S1_SELECT sfn_nw=%d' % state['sfn_nw'], flush=True)


_s1_prev_prepare = prepare


def prepare(weights):
 t0 = __import__('time').perf_counter()
 state = _s1_prev_prepare(weights)
 state['sfn_nw'] = 8
 if __import__('time').perf_counter() - t0 > 285.0:
  print('S1_SKIP', flush=True)
  return state
 try:
  _s1_select(state)
 except Exception as e:
  state['sfn_nw'] = 8
  print('S1_ERROR', repr(e)[:300], flush=True)
 return state
'''


def main():
    src = (HERE / 'decoder_r2.py').read_bytes().decode('utf-8')
    assert hashlib.sha256(src.encode()).hexdigest().startswith('1f601e3b'), 'decoder_r2.py must be the scored R2 file'
    assert src.count(EDIT[0]) == 1
    out = src.replace(EDIT[0], EDIT[1]).rstrip('\n') + '\n' + APPEND
    ast.parse(out)
    b = out.encode('utf-8')
    (HERE / 'decoder_s1.py').write_bytes(b)
    print(json.dumps({'sha256': hashlib.sha256(b).hexdigest(), 'bytes': len(b)}))


if __name__ == '__main__':
    main()
