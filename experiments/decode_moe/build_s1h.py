"""S1 rebuilt on the shared harness (experiments/decode_moe/harness.py): same selection rule as build_s1.py.

Parent: experiments/decode_moe/decoder_r2.py (scored 2.0339x). Output decoder_s1h.py.
"""
import ast
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from harness import append_harness  # noqa: E402
from build_s1 import EDIT  # noqa: E402

APPEND = r'''

# S1h: _sfn_kernel warps A/B on the shared harness; bitwise + > 3% at both T = 1 and 8.
def _s1_select(state):
 if state.get('sliced_ffn_boundary') is None or state.get('rf') is None:
  return
 dev = state['rf']['weight'].device
 med_all, ok_all = {}, {}
 for T in (1, 8):
  g = torch.Generator(device=dev).manual_seed(77 + T)
  br = torch.randn(T, UNPOOL_SLICE, device=dev, generator=g).to(torch.bfloat16)
  st = torch.randn(T, HIDDEN, device=dev, generator=g)
  ref = None
  fns, ok = {}, {}
  for nw in (8, 4, 16, 32):
   state['sfn_nw'] = nw
   out = [t.clone() for t in _sfn_boundary(st, br, state)]
   ref = out if nw == 8 else ref
   ok[nw] = _okh_same(ref, out)
   fns[nw] = (lambda n: (lambda: (state.__setitem__('sfn_nw', n), _sfn_boundary(st, br, state))))(nw)
  med = _okh_time(fns, dev, reps=30)
  state['sfn_nw'] = 8
  med_all[T], ok_all[T] = med, ok
  print('S1_SFN T=%d ' % T + ' '.join('nw%d=%.2fus%s' % (n, med[n], '' if ok[n] else '(differs)') for n in med), flush=True)
 picks = {T: _okh_pick(med_all[T], 8, allowed=[n for n in (4, 16, 32) if ok_all[T][n]]) for T in (1, 8)}
 state['sfn_nw'] = picks[1] if picks[1] == picks[8] else 8
 print('S1_SELECT sfn_nw=%d' % state['sfn_nw'], flush=True)


_s1_prev_prepare = prepare


def prepare(weights):
 t0 = _okh_clock.perf_counter()
 state = _s1_prev_prepare(weights)
 state['sfn_nw'] = 8
 if not _okh_budget(t0, 285.0):
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
    assert hashlib.sha256(src.encode()).hexdigest().startswith('1f601e3b')
    assert src.count(EDIT[0]) == 1
    out = append_harness(src.replace(EDIT[0], EDIT[1])) + APPEND
    ast.parse(out)
    b = out.encode('utf-8')
    (HERE / 'decoder_s1h.py').write_bytes(b)
    print(json.dumps({'sha256': hashlib.sha256(b).hexdigest(), 'bytes': len(b)}))


if __name__ == '__main__':
    main()
