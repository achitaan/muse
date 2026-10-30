"""WQ: runtime stream-K range quantum for the MoE WGS build, plus a MoE-track prepare-time profiler (Phase 1).

CUDA change (WGS only; every other build compiles identical device code apart from one unused Params field):
  * Params.wq (from the high bits of okc_run's `grid` argument: grid | qa << 16 | qb << 24; existing callers pass
    a plain grid, so wq = 0 and the parent's ranges are used unchanged).
  * qa / qb > 1 round each CTA's phase-A / phase-B range bounds to multiples of q stages (q divides 16), so
    stream-K splits land on q-stage boundaries (q = 16: whole 256-row units, no fixups). The fixup-slot owner
    lookup (wgs_cta_of) uses the same rounded bounds.
The MoE-track candidate keeps the incumbent timed path (kernel_fn unchanged, wq = 0). Its prepare() runs a
profiler on Decode-like routing (draws from the muse_decode_layer reference routing model) and MoE-track
routing: every variant (q values, grid sizes, MAIN vs MID at 16 tokens) is checked against the incumbent output
and timed as interleaved cold-L2 CUDA-graph replays. Nothing is selected; the log is the experiment.
"""
import ast
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OPUS = HERE.parents[2] / 'opus-fresh' / 'experiments' / 'opus_fresh'
PARENT = OPUS / 'candidates' / 'moe_finreg.py'
sys.path.insert(0, str(OPUS))

CUDA_EDITS = [
    ("""__device__ void wgs_ranges(const Params& p, Smem& S);""", """__device__ void wgs_ranges(int wq, Smem& S);"""),
    ("""if ((threadIdx.x & 31) == 0) wgs_ranges(p, S);""", """if ((threadIdx.x & 31) == 0) wgs_ranges(p.wq, S);"""),
    ("""  int wgr[4];         // WGS: this CTA's phase A and phase B unit ranges
""", """  int wgr[4];         // WGS: this CTA's phase A and phase B unit ranges
  int wown[2];        // WQ: per phase, the CTA holding the first stage of the unit containing wgr[2 pb]
"""),
    ("""  int M, lo, nloc;
""", """  int M, lo, nloc;
  int wq;          // WQ: stream-K range quantum, phase A bits 0..7, phase B bits 8..15 (0 or 1: parent ranges)
"""),
    ("""// this CTA's stage ranges (stream-K over 256-row units x 16 stages), one thread: the producers stream the same
__device__ __noinline__ void wgs_ranges(const Params& p, Smem& S) {
  const int nj = S.njF + S.njS;
  wgs_prefix(S, nj);
  wgs_range(S, nj, 0, blockIdx.x, gridDim.x, S.wgr[0], S.wgr[1]);
  wgs_range(S, nj, 1, blockIdx.x, gridDim.x, S.wgr[2], S.wgr[3]);
}
// the CTA whose range holds stage s (the fixup slot of a split unit is the CTA holding its first stage)
__device__ __forceinline__ int wgs_cta_of(const Smem& S, int pb, int s, int G) {
  const uint32_t tot = pb ? S.wtotB : S.wtot;""",
     """__device__ __forceinline__ int wgs_cta_of(const Smem& S, int pb, int s, int G);  // (defined below)
// WQ: range bound c (0..G) rounded to a multiple of q stages; q divides 16, so bound(G) = nj << (9 - pb) is exact
__device__ __forceinline__ int wgs_q(int wq, int pb) {
  const int q = (wq >> (8 * pb)) & 31;
  return q > 1 ? q : 1;
}
__device__ __forceinline__ int wgs_bound(const Smem& S, int nj, int pb, int c, int G, int q) {
  if (c >= G) return nj << (9 - pb);
  const uint32_t tot = pb ? S.wtotB : S.wtot;
  const int g = wgs_stage_at(S, nj, pb, tot * (uint32_t)c / (uint32_t)G);
  return (g + (q >> 1)) / q * q;
}
// this CTA's stage ranges (stream-K over 256-row units x 16 stages), one thread: the producers stream the same
// (WQ: takes the quantum by value; a Params reference into a non-inlined call copies Params to local memory)
__device__ __noinline__ void wgs_ranges(int wq, Smem& S) {
  const int nj = S.njF + S.njS;
  wgs_prefix(S, nj);
#pragma unroll 1
  for (int pb = 0; pb < 2; ++pb) {
    const int q = wgs_q(wq, pb);
    if (q == 1) {
      wgs_range(S, nj, pb, blockIdx.x, gridDim.x, S.wgr[2 * pb], S.wgr[2 * pb + 1]);
      // the only share that can start mid-unit is the one at the range start: precompute its fixup slot here
      S.wown[pb] = wgs_cta_of(S, pb, (S.wgr[2 * pb] >> 4) << 4, gridDim.x);
    } else {
      const int g0 = wgs_bound(S, nj, pb, blockIdx.x, gridDim.x, q);
      S.wgr[2 * pb] = g0;
      S.wgr[2 * pb + 1] = wgs_bound(S, nj, pb, blockIdx.x + 1, gridDim.x, q);
      // a share starting mid-unit can only be the one at g0: its fixup slot is the CTA holding the unit's first
      // stage, the largest c with bound(c) <= that stage (empty ranges are skipped)
      const int s = (g0 >> 4) << 4;
      int lo = 0, hi = gridDim.x;
      while (hi - lo > 1) {
        const int mid = (lo + hi) >> 1;
        if (wgs_bound(S, nj, pb, mid, gridDim.x, q) <= s) lo = mid; else hi = mid;
      }
      S.wown[pb] = lo;
    }
  }
}
// the CTA whose range holds stage s (the fixup slot of a split unit is the CTA holding its first stage)
__device__ __forceinline__ int wgs_cta_of(const Smem& S, int pb, int s, int G) {
  const uint32_t tot = pb ? S.wtotB : S.wtot;"""),
    ("""wgs_cta_of(S, PB, u << 4, gridDim.x)""", """S.wown[PB]"""),
    ("""  p.M = (int)x.size(0); p.lo = (int)lo; p.nloc = (int)nloc;
""", """  p.M = (int)x.size(0); p.lo = (int)lo; p.nloc = (int)nloc;
  p.wq = (int)((grid >> 16) & 0xffff);  // WQ: optional range quantum in the high bits of grid
  grid &= 0xffff;
"""),
]

HARNESS = r'''

# ---------------------------------------------------------------- WQ profiler (decode-moe-plan Phase 1)
import time as _wq_time
_WQ_VARIANTS = (("mainq1_16", 1, 16, 0), ("mainq1_8", 1, 8, 0), ("mainq8_8", 8, 8, 0), ("mainq4_4", 4, 4, 0),
                ("mainq8_16", 8, 16, 0), ("mainq16_16", 16, 16, 0), ("main_g120", 0, 0, 120),
                ("main_g100", 0, 0, 100), ("main_g66", 0, 0, 66))


def _wq_cases():
    out = []
    for name, (T, flat) in _WQ_IDS.items():
        out.append((name, T, flat))
    return out


def _wq_profile(state, reps=30):
    dev = state["w"][0].device
    sfx = state.get("sfx", ())
    main = _ext(tuple(_main_defines(sfx)) + sfx)
    mid = _ext(tuple(MID_DEFINES) + sfx)
    G = state["grid"]
    flush = torch.empty(256 << 20, dtype=torch.uint8, device=dev)
    summary = {}
    for name, T, flat in _wq_cases():
        g = torch.Generator(device=dev).manual_seed(T + len(name))
        x = (torch.randn(T, HIDDEN, device=dev, generator=g) * 0.5).to(torch.bfloat16)
        ids = torch.tensor(flat, dtype=torch.int32, device=dev).view(T, TOP_K).contiguous()
        tw = torch.rand(T, TOP_K, device=dev, generator=g) + 0.05
        tw = (tw / tw.sum(-1, keepdim=True)).float().contiguous()
        inc = mid if T <= MID_MAX_M else main
        vs = [("inc", inc, G)] + ([("main", main, G)] if T <= MID_MAX_M else [])
        for lab, qa, qb, gr in _WQ_VARIANTS:
            vs.append((lab, main, (gr or G) | (qa << 16) | (qb << 24)))
        outs, graphs = {}, {}
        for lab, ext, garg in vs:
            o = torch.empty(T, HIDDEN, dtype=torch.bfloat16, device=dev)
            ext.okc_run(state["w"], state["scratch"], x, ids, tw, o, state["lo"], state["cnt"], garg)
            torch.cuda.synchronize()
            outs[lab] = o.clone()
            gph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gph):
                ext.okc_run(state["w"], state["scratch"], x, ids, tw, o, state["lo"], state["cnt"], garg)
            graphs[lab] = gph
        ref = outs["inc"].float()
        errs = {}
        for lab, o in outs.items():
            d = o.float() - ref
            errs[lab] = float(d.norm() / ref.norm().clamp_min(1e-30)) if bool(torch.isfinite(o.float()).all()) else float("inf")
        evs = []
        for _ in range(reps):
            for lab, gph in graphs.items():
                flush.zero_()
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record(); gph.replay(); b.record()
                evs.append((lab, a, b))
        torch.cuda.synchronize()
        times = {lab: [] for lab in graphs}
        for lab, a, b in evs:
            times[lab].append(a.elapsed_time(b) * 1000.0)
        med = {lab: sorted(t)[len(t) // 2] for lab, t in times.items()}
        cnt = torch.bincount(ids.flatten().long(), minlength=256)[:32]
        print(f"WQ_CASE {name} T={T} local_routes={int(cnt.sum())} experts={int((cnt > 0).sum())} max={int(cnt.max())}",
              flush=True)
        for lab in graphs:
            print(f"WQ {name} {lab:12s} {med[lab]:8.2f}us ratio={med[lab] / med['inc']:.4f} relL2={errs[lab]:.2e}", flush=True)
            summary.setdefault((name.split('_')[0], T), {}).setdefault(lab, []).append(med[lab] / med["inc"])
        del graphs
    for key, d in summary.items():
        print(f"WQ_SUMMARY {key[0]} T={key[1]} " + " ".join(
            f"{lab}={sorted(v)[len(v) // 2]:.4f}" for lab, v in d.items()), flush=True)


_wq_prev_prepare = prepare


def prepare(weights):
    t0 = _wq_time.perf_counter()
    state = _wq_prev_prepare(weights)
    try:
        if torch.cuda.get_device_capability()[0] == 9 and state.get("sfx"):
            _wq_profile(state)
    except Exception as e:  # the experiment never affects the timed path
        print("WQ_ERROR", repr(e), flush=True)
    torch.cuda.synchronize()
    print(f"WQ_DONE {_wq_time.perf_counter() - t0:.1f}s", flush=True)
    return state
'''


def draw_ids():
    import torch
    import routing_comparison as rc

    def dec(batch, q, seed):
        tokens = batch * q
        # replicate decode_track_sample but keep the ids (rotated so the busiest 32-expert pair is local)
        import math
        s_gen = torch.Generator().manual_seed(rc.STREAM_SEED)
        mean = torch.randn((rc.HIDDEN,), generator=s_gen)
        shared = torch.randn((rc.STREAM_RANK, rc.HIDDEN), generator=s_gen)
        r_gen = torch.Generator().manual_seed(5353 + seed * 17)
        w = rc.ROUTER_STD * torch.randn((rc.N_EXPERTS, rc.HIDDEN), generator=r_gen)
        align = rc.ROUTER_ALIGN * torch.randn((rc.N_EXPERTS, 1), generator=r_gen)
        router = w + align * mean[None, :] / math.sqrt(rc.HIDDEN)
        bias = rc.ROUTER_BIAS_STD * torch.randn((rc.N_EXPERTS,), generator=r_gen)
        pre_ffn = 0.1 * torch.randn((rc.HIDDEN,), generator=r_gen)
        gen = torch.Generator().manual_seed(seed)
        coef = torch.randn((tokens, rc.STREAM_RANK), generator=gen)
        scale = torch.ones((batch, q, 1))
        if 1 < q <= 8:
            scale[:, 1:] = rc.DRAFT_SHARE
        scale = scale.reshape(tokens, 1)
        stream = (scale * (rc.STREAM_MEAN * mean + math.sqrt(rc.STREAM_SHARED / rc.STREAM_RANK) * coef @ shared)
                  + (1 - rc.STREAM_SHARED * scale * scale).sqrt() * torch.randn((tokens, rc.HIDDEN), generator=gen))
        h = stream * torch.rsqrt(stream.pow(2).mean(-1, keepdim=True) + rc.RMS_EPS) * (1.0 + pre_ffn)
        ids = (torch.sigmoid((h @ router.T) * rc.ROUTER_MULTIPLIER) + bias).topk(rc.TOP_K, dim=-1).indices
        pairs = torch.bincount((ids // 32).flatten(), minlength=8)
        return ((ids - 32 * int(pairs.argmax())) % 256).flatten().tolist()

    def moe(bs, seed):
        gen = torch.Generator(device="cpu").manual_seed(seed)
        scores = torch.rand((bs, 256), generator=gen)
        z = torch.randn((256,), generator=gen)
        u = torch.rand((256,), generator=gen).clamp(min=1e-12)
        scores = scores + rc.POPULARITY * z / (-u.log()).sqrt()
        ids = scores.topk(8, dim=-1).indices
        pairs = torch.bincount((ids // 32).flatten(), minlength=8)
        return ((ids - 32 * int(pairs.argmax())) % 256).flatten().tolist()

    d = {}
    for s in (1, 2, 3):
        d[f"dec16_{s}"] = (16, dec(16, 1, 4000 + s))
        d[f"dec32_{s}"] = (32, dec(32, 1, 4100 + s))
    for s in (1, 2):
        d[f"moe16_{s}"] = (16, moe(16, 4200 + s))
        d[f"moe32_{s}"] = (32, moe(32, 4300 + s))
    d["dec128_1"] = (128, dec(16, 8, 4400))
    d["moe128_1"] = (128, moe(128, 4500))
    return d


def main():
    src = PARENT.read_bytes().decode('utf-8')
    assert hashlib.sha256(src.encode()).hexdigest().startswith('fe72a4b2')
    t = ast.parse(src)
    n = next(n for n in t.body if isinstance(n, ast.Assign) and getattr(n.targets[0], 'id', '') == 'CUDA_SRC')
    seg = ast.get_source_segment(src, n.value)
    assert src.count(seg) == 1
    cuda = n.value.value
    for old, new in CUDA_EDITS:
        assert cuda.count(old) == 1, old[:70]
        cuda = cuda.replace(old, new)
    out = src.replace(seg, repr(cuda))
    assert out.count('okc_moe_w4a16_v216x_finreg1_') == 1
    out = out.replace('okc_moe_w4a16_v216x_finreg1_', 'okc_moe_w4a16_v216x_wq1_')
    ids = draw_ids()
    out = out.rstrip('\n') + '\n\n_WQ_IDS = ' + repr({k: (T, v) for k, (T, v) in ids.items()}) + '\n' + HARNESS
    ast.parse(out)
    (HERE / 'moe_wq.cu').write_text(cuda, encoding='utf-8', newline='')
    (HERE / 'moe_wq.py').write_bytes(out.encode('utf-8'))
    stats = {}
    import torch
    for k, (T, v) in ids.items():
        c = torch.bincount(torch.tensor(v), minlength=256)[:32]
        stats[k] = (int(c.sum()), int((c > 0).sum()), int(c.max()))
    print(json.dumps({'sha256': hashlib.sha256(out.encode()).hexdigest(), 'bytes': len(out.encode()),
                      'local_routes/experts/max': stats}, indent=1))


if __name__ == '__main__':
    main()
