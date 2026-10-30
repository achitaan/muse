"""Shared prepare-time A/B harness for Decode/MoE candidates (decode-moe-plan Phase 0).

Builders append HARNESS_SRC (once) to a candidate source and call these helpers from their own prepare
wrappers instead of copying the logic. Everything runs during prepare() on the judge GPU and never changes
the timed path by itself; a candidate decides what to enable from the returned numbers.

  _okh_same(ref, out, rtol=0.0)       bitwise equality of tensor tuples (or relative tolerance per tensor)
  _okh_time(fns, dev, reps=25)        interleaved cold-L2 CUDA-graph timing; fns = {label: zero-arg callable}
                                      returns {label: median microseconds}; graphs are captured after one
                                      eager warm-up call each and freed before returning
  _okh_pick(med, base, margin=0.97)   fastest label beating med[base] by more than (1 - margin), else base
  _okh_budget(t0, limit_s)            True while elapsed prepare time is below limit_s (the judge caps at 300 s)
  _okh_bg(fn)                         start fn on a daemon thread (e.g. an extension build) and return the thread
"""

HARNESS_SRC = r'''

# ---- shared prepare-time A/B harness (experiments/decode_moe/harness.py)
import threading as _okh_threading
import time as _okh_clock


def _okh_same(ref, out, rtol=0.0):
 if out is None or len(ref) != len(out):
  return False
 for a, b in zip(ref, out):
  if a.shape != b.shape or a.dtype != b.dtype:
   return False
  if rtol == 0.0:
   if not torch.equal(a, b):
    return False
  elif not bool(((a.float() - b.float()).abs() <= rtol * a.float().abs()).all()):
   return False
 return True


def _okh_time(fns, dev, reps=25):
 flush = torch.empty(256 << 20, dtype=torch.uint8, device=dev)
 graphs = {}
 for lab, fn in fns.items():
  fn()
  torch.cuda.synchronize()
  gr = torch.cuda.CUDAGraph()
  with torch.cuda.graph(gr):
   fn()
  graphs[lab] = gr
 evs = []
 for _ in range(reps):
  for lab, gr in graphs.items():
   flush.zero_()
   a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
   a.record(); gr.replay(); b.record()
   evs.append((lab, a, b))
 torch.cuda.synchronize()
 ts = {lab: [] for lab in graphs}
 for lab, a, b in evs:
  ts[lab].append(a.elapsed_time(b) * 1000.0)
 del graphs
 return {lab: sorted(v)[len(v) // 2] for lab, v in ts.items()}


def _okh_pick(med, base, margin=0.97, allowed=None):
 cand = {k: v for k, v in med.items() if k != base and (allowed is None or k in allowed)}
 if not cand or base not in med:
  return base
 best = min(cand, key=cand.get)
 return best if cand[best] < margin * med[base] else base


def _okh_budget(t0, limit_s):
 return _okh_clock.perf_counter() - t0 < limit_s


def _okh_bg(fn):
 th = _okh_threading.Thread(target=fn, daemon=True)
 th.start()
 return th
'''


def append_harness(src: str) -> str:
    """Append the harness once (idempotent)."""
    if '# ---- shared prepare-time A/B harness' in src:
        return src
    return src.rstrip('\n') + '\n' + HARNESS_SRC
