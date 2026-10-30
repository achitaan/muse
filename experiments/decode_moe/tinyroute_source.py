"""Exact small-route table construction; no changes to expert arithmetic."""

import ast
import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = ROOT / 'results/decode-moe-plan-20261030/tinyroute'
BASE_SHA = '331148716e8d1ba92b45cd29725be997d44b3a5235af74d75770bb9444100d50'

PREAMBLE = '''
#ifndef OKC_ROUTE_WARP
#define OKC_ROUTE_WARP 0
#endif
#if OKC_ROUTE_WARP
#define moe_fused moe_tr8_fused
#endif
'''

# Each warp lane represents an expert for counts, and a route within each
# 32-entry chunk for positions. Five ballot bitplanes transpose those roles.
# Two chunks cover every route at M <= 8; the prefix keeps expert order and
# the match masks keep ascending token*8+slot within each expert.
FAST_ROUTE = '''
#if OKC_ROUTE_WARP && FAST1_ON && !OKC_SPLIT && !OKC_WGS
  if (p.M <= 8 && p.nloc <= 32) {
    if (warp == 0) {
      int e0 = lane < E ? __ldg(p.topk_ids + lane) - p.lo : -1;
      int e1 = lane + 32 < E ? __ldg(p.topk_ids + lane + 32) - p.lo : -1;
      e0 = e0 >= 0 && e0 < p.nloc ? e0 : -1;
      e1 = e1 >= 0 && e1 < p.nloc ? e1 : -1;
      unsigned own0 = __ballot_sync(0xffffffffu, e0 >= 0);
      unsigned own1 = __ballot_sync(0xffffffffu, e1 >= 0);
#pragma unroll
      for (int bit = 0; bit < 5; ++bit) {
        const unsigned plane0 = __ballot_sync(0xffffffffu, (e0 & (1 << bit)) != 0);
        const unsigned plane1 = __ballot_sync(0xffffffffu, (e1 & (1 << bit)) != 0);
        own0 &= lane & (1 << bit) ? plane0 : ~plane0;
        own1 &= lane & (1 << bit) ? plane1 : ~plane1;
      }
      const int c0 = __popc(own0), count = c0 + __popc(own1);
      const bool supported = __all_sync(0xffffffffu, count <= SK_T);
      // Unexpected repeated-slot inputs with fat jobs retain the generic
      // implementation; publish the fallback decision to all compute warps.
      if (lane == 0) S.njF = supported ? 0 : -1;
      if (supported) {
        int incl = count;
#pragma unroll
        for (int d = 1; d < 32; d <<= 1) {
          const int v = __shfl_up_sync(0xffffffffu, incl, d);
          if (lane >= d) incl += v;
        }
        const int off = incl - count;
        const unsigned present = __ballot_sync(0xffffffffu, count != 0);
        const unsigned before = (1u << lane) - 1u;
        S.cnt[lane] = count;
        S.off[lane] = off;
        if (count) {
          const int job = __popc(present & before);
          S.jE[job] = lane; S.jS[job] = off; S.jC[job] = count;
        }
        if (lane == 0) S.njS = __popc(present);
        const unsigned peers0 = __match_any_sync(0xffffffffu, e0);
        const unsigned peers1 = __match_any_sync(0xffffffffu, e1);
        const int off0 = __shfl_sync(0xffffffffu, off, e0 >= 0 ? e0 : 0);
        const int off1 = __shfl_sync(0xffffffffu, off, e1 >= 0 ? e1 : 0);
        const int prior1 = __shfl_sync(0xffffffffu, c0, e1 >= 0 ? e1 : 0);
        if (lane < E) {
          S.eid[lane] = (signed char)e0;
          const int q = e0 >= 0 ? off0 + __popc(peers0 & before) : -1;
          S.pos[lane] = (short)q;
          if (q >= 0) S.ent[q] = (short)lane;
        }
        if (lane + 32 < E) {
          S.eid[lane + 32] = (signed char)e1;
          const int q = e1 >= 0 ? off1 + prior1 + __popc(peers1 & before) : -1;
          S.pos[lane + 32] = (short)q;
          if (q >= 0) S.ent[q] = (short)(lane + 32);
        }
      }
    }
    // Includes only the original NT compute threads (helper joins later).
    // All table writes precede this barrier. FAST1 already owns early_issue.
    CTA_SYNC();
    if (S.njF == 0) return;
  }
#endif
'''


def derive():
    raw = (ROOT / 'best/moe_verified.py').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == BASE_SHA
    tree = ast.parse(raw)
    nodes = {n.targets[0].id: n.value for n in tree.body
             if isinstance(n, ast.Assign) and len(n.targets) == 1
             and isinstance(n.targets[0], ast.Name)}
    parent = ast.literal_eval(nodes['CUDA_SRC'])
    anchor = '  const int E = p.M * 8;\n'
    assert parent.count(anchor) == 1
    modified = PREAMBLE + parent.replace(anchor, anchor + FAST_ROUTE)
    assert modified.removeprefix(PREAMBLE).replace(FAST_ROUTE, '', 1) == parent
    flags = {name: ast.literal_eval(nodes[name + '_DEFINES']) for name in ['TINY', 'TINY2', 'MID', 'MAIN']}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / 'parent.cu').write_bytes(parent.encode())
    (RESULTS / 'candidate.cu').write_bytes(modified.encode())
    return raw, parent, modified, flags
