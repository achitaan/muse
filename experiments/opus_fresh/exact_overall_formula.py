"""Exact reverse-engineering of the archived overall leaderboard ranking.

Purely offline analysis of results/model-handoff-20261029/*.json.
"""
import json, math
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SNAP = ROOT / "results/model-handoff-20261029"
OV = ["muse_bf16_proj", "muse_attention_decode", "muse_moe_w4a16", "muse_decode_layer"]


def load():
    boards = {}
    for t in OV + ["muse_prefill_layer"]:
        r = json.loads((SNAP / f"{t}-board.json").read_text())["response"]
        boards[t] = {row["team"]: row for row in r["rows"]}
    ov = json.loads((SNAP / "overall.json").read_text())["response"]
    return boards, ov


def main():
    boards, ov = load()
    rows = ov["rows"]
    dec = boards["muse_decode_layer"]

    # 1. Check whether among teams that ONLY solved muse_decode_layer (or where decode dominates),
    # their order is strictly decode_speedup (or decode value).
    only_dec = [r for r in rows if r["solved"] == 1 and r["team"] in dec]
    print("solved==1 & in decode:", len(only_dec))

    #Use all teams whose only entry is decode as exact rulers!
    rulers = [(r["rank"], dec[r["team"]]["speedup"], dec[r["team"]]["value"], r["team"])
              for r in rows if r["solved"] == 1 and r["team"] in dec]
    print("rulers count:", len(rulers), "first 5:", rulers[:5])

    # Wait: what about teams in ranks 1..314? Every one of them is in decode and has rank == decode.rank!
    # Check if there is ANY inversion in decode["rank"] among ALL 977 decode teams!
    dec_ranks_in_ov = [dec[r["team"]]["rank"] for r in rows if r["team"] in dec]
    inv = [(i, dec_ranks_in_ov[i], dec_ranks_in_ov[i+1])
           for i in range(len(dec_ranks_in_ov)-1) if dec_ranks_in_ov[i] > dec_ranks_in_ov[i+1]]
    print("inversions of decode.rank among decode teams in overall:", len(inv))
    # Identify "backbone" decode teams: those whose decode.rank is strictly increasing.
    # Since non-decode composite can only INCREASE a team's effective speedup (move it earlier),
    # a decode team is on the backbone iff its decode.rank > max decode.rank of all LATER decode teams?
    # No: moving earlier means a team with LARGE decode.rank (low decode speedup) appears BEFORE teams
    # with SMALLER decode.rank. So a decode team has NOT jumped iff no later decode team has a smaller decode.rank!
    dec_entries = [(idx, r["team"], dec[r["team"]]["rank"], dec[r["team"]]["speedup"], dec[r["team"]]["value"])
                   for idx, r in enumerate(rows) if r["team"] in dec]
    min_later_rank = [10**9] * len(dec_entries)
    cur = 10**9
    for i in range(len(dec_entries) - 1, -1, -1):
        min_later_rank[i] = cur
        if dec_entries[i][2] < cur:
            cur = dec_entries[i][2]
    backbone = {}
    jumped_dec = []
    for (idx, t, drank, dsp, dval), mlr in zip(dec_entries, min_later_rank):
        if drank < mlr:
            backbone[idx] = (t, drank, dsp, dval)
        else:
            jumped_dec.append((idx, t, drank, dsp, dval))
    print(f"backbone decode teams: {len(backbone)}, jumped decode teams: {len(jumped_dec)}")
    # Verify backbone is strictly increasing in drank and decreasing in dsp
    bb_ranks = [v[1] for k, v in sorted(backbone.items())]
    assert all(a < b for a, b in zip(bb_ranks, bb_ranks[1:])), "backbone not strictly increasing!"
    print("backbone is 100% strictly ordered by decode.rank!")

    # Now every non-backbone team (no-decode + jumped_dec) has an EXACT interval [lo_sp, hi_sp]
    # from the nearest backbone neighbours!
    intervals = []
    bb_indices = sorted(backbone.keys())
    import bisect
    for idx, r in enumerate(rows):
        if idx in backbone:
            continue
        t = r["team"]
        pos = bisect.bisect_left(bb_indices, idx)
        hi_sp = backbone[bb_indices[pos - 1]][2] if pos > 0 else 10.0
        lo_sp = backbone[bb_indices[pos]][2] if pos < len(bb_indices) else 1.0
        hi_val = backbone[bb_indices[pos - 1]][3] if pos > 0 else 0.0
        lo_val = backbone[bb_indices[pos]][3] if pos < len(bb_indices) else 30000.0
        sb = boards["muse_bf16_proj"][t]["speedup"] if t in boards["muse_bf16_proj"] else 1.0
        sa = boards["muse_attention_decode"][t]["speedup"] if t in boards["muse_attention_decode"] else 1.0
        sm = boards["muse_moe_w4a16"][t]["speedup"] if t in boards["muse_moe_w4a16"] else 1.0
        vb = boards["muse_bf16_proj"][t]["value"] if t in boards["muse_bf16_proj"] else 4830.8005
        va = boards["muse_attention_decode"][t]["value"] if t in boards["muse_attention_decode"] else 3766.3049
        vm = boards["muse_moe_w4a16"][t]["value"] if t in boards["muse_moe_w4a16"] else 13448.4945
        intervals.append((idx + 1, t, lo_sp, hi_sp, lo_val, hi_val, sb, sa, sm, vb, va, vm))

    print("non-backbone intervals:", len(intervals))
    for x in intervals[:10]:
        print(f"  ov {x[0]:4d} {x[1]:16s} sp in [{x[2]:.4f}, {x[3]:.4f}] val in [{x[5]:.1f}, {x[4]:.1f}]"
              f" sb={x[6]:.4f} sa={x[7]:.4f} sm={x[8]:.4f}")

    # Linear regression / LP: if effective decode time is T_eff = c0 + cb*vb + ca*va + cm*vm
    # or 1/S_eff = w0 + wb/sb + wa/sa + wm/sm, test whether a linear function of (1/sb, 1/sa, 1/sm)
    # lies inside [1/hi_sp, 1/lo_sp] for ALL 733 intervals!
    A = np.array([[1.0, 1.0/x[6], 1.0/x[7], 1.0/x[8]] for x in intervals])
    y_mid = np.array([0.5 * (1.0/x[2] + 1.0/x[3]) for x in intervals])
    tight = np.array([(1.0/x[2] - 1.0/x[3]) < 0.02 for x in intervals])
    w, *_ = np.linalg.lstsq(A[tight], y_mid[tight], rcond=None)
    print("fitted w (w0, wb, wa, wm):", w, "sum:", w.sum())
    pred_sp = 1.0 / (A @ w)
    viol = 0
    for (rk, t, lo, hi, *_), p in zip(intervals, pred_sp):
        if p < lo - 5e-4 or p > hi + 5e-4:
            viol += 1
    print(f"harmonic linear violations: {viol} / {len(intervals)}")
    # Compare against actual SGLang per-load baseline times from the 4 saved profile JSONs!
    profs = {
        "muse_bf16_proj": ROOT / "results/sprint-20261028-evidence/6131bc96427e4486/profiles/submission.6131bc96427e4486.json",
        "muse_attention_decode": ROOT / "results/attention-segment-repair-20261028/publication/9e922c3555484081ca45a99a90284e26eccab181fb677700d8227a4b3402412f/profiles/submission.8b767ea7b2b74de6.json",
        "muse_moe_w4a16": ROOT / "results/resumed-moe-phase4-20261028/331148716e8d1ba92b45cd29725be997d44b3a5235af74d75770bb9444100d50/profiles/submission.d600231c8a8244cb.json",
        "muse_decode_layer": ROOT / "results/decoder-segment-repair-20261028/profiles/submission.9560c915e85f4f50.json",
    }
    loads = ["decode 1", "decode 16", "decode 32", "verify 1", "verify 16", "verify 32"]
    sg_load = {k: {L: 0.0 for L in loads} for k in profs}
    for k, p in profs.items():
        d = json.loads(p.read_text())
        for b in d["benchmarks"]:
            c = b["case"]
            sg_load[k][c["load"]] += c["weight"] * (b["baseline_ns"] / 1e3)
    print("\nSGLang per-load baseline times (us) from saved profiles:")
    ratios = {k: [] for k in ("muse_bf16_proj", "muse_attention_decode", "muse_moe_w4a16", "rest")}
    for L in loads:
        td = sg_load["muse_decode_layer"][L]
        tb, ta, tm = sg_load["muse_bf16_proj"][L], sg_load["muse_attention_decode"][L], sg_load["muse_moe_w4a16"][L]
        #Wait: in muse_moe_w4a16 weight is 60, in decode_layer there are 62 layers! What if MoE is scaled or unscaled?
        tr = td - (tb + ta + tm)
        ratios["muse_bf16_proj"].append(tb / td)
        ratios["muse_attention_decode"].append(ta / td)
        ratios["muse_moe_w4a16"].append(tm / td)
        ratios["rest"].append(tr / td)
        print(f"  {L:10s}: dec={td:8.1f}  bf16={tb:7.1f} ({tb/td:.3f})  attn={ta:7.1f} ({ta/td:.3f})"
              f"  moe={tm:8.1f} ({tm/td:.3f})  rest={tr:7.1f} ({tr/td:.3f})")
    # Use unrounded speedups s_k = B_k / val_k (16-digit float) and test constrained w0+wb+wa+wm = 1
    B = {"muse_bf16_proj": 4830.80049961482, "muse_attention_decode": 3766.3048835752334,
         "muse_moe_w4a16": 13448.494516802382, "muse_decode_layer": 22620.31148392537}
    intervals_u = []
    for idx, r in enumerate(rows):
        if idx in backbone:
            continue
        t = r["team"]
        pos = bisect.bisect_left(bb_indices, idx)
        hi_sp = B["muse_decode_layer"] / backbone[bb_indices[pos - 1]][3] if pos > 0 else 10.0
        lo_sp = B["muse_decode_layer"] / backbone[bb_indices[pos]][3] if pos < len(bb_indices) else 1.0
        sb = B["muse_bf16_proj"] / boards["muse_bf16_proj"][t]["value"] if t in boards["muse_bf16_proj"] else 1.0
        sa = B["muse_attention_decode"] / boards["muse_attention_decode"][t]["value"] if t in boards["muse_attention_decode"] else 1.0
        sm = B["muse_moe_w4a16"] / boards["muse_moe_w4a16"][t]["value"] if t in boards["muse_moe_w4a16"] else 1.0
        intervals_u.append((idx + 1, t, lo_sp, hi_sp, sb, sa, sm))

    # Solve LP / minimax for w = (w0, wb, wa, wm) to minimize max interval violation!
    from scipy.optimize import minimize
    lo_inv = np.array([1.0 / x[3] for x in intervals_u])  # 1/hi_sp <= 1/S <= 1/lo_sp
    hi_inv = np.array([1.0 / x[2] for x in intervals_u])
    Au = np.array([[1.0, 1.0/x[4], 1.0/x[5], 1.0/x[6]] for x in intervals_u])
    def loss(w):
        v = Au @ w
        err = np.maximum(0.0, np.maximum(lo_inv - v, v - hi_inv))
        return (err ** 2).sum() + 1e-4 * err.max()
    res = minimize(loss, w, method="Nelder-Mead", options={"maxiter": 20000, "xatol": 1e-10, "fatol": 1e-14})
    w_opt = res.x
    v = Au @ w_opt
    err = np.maximum(0.0, np.maximum(lo_inv - v, v - hi_inv))
    print("w_opt:", w_opt, "sum:", w_opt.sum(), "max_inv_viol:", err.max(),
          "num viol > 1e-5:", (err > 1e-5).sum(), "num viol > 1e-4:", (err > 1e-4).sum())
    # Also test per-load geometric mean model: what if each load L has weights w_{L,0}, w_{L,b}, w_{L,a}, w_{L,m}?
    # Under track-level speedups only (the board only stores track-level value/speedup, or does it?),
    # if the backend only uses track-level speedups, a single harmonic mean vs per-load geo-mean of harmonic means:
    # Test per-load combination on our own saved benchmark JSONs!
    our_load = {k: {L: 0.0 for L in loads} for k in profs}
    for k, p in profs.items():
        d = json.loads(p.read_text())
        for b in d["benchmarks"]:
            c = b["case"]
            our_load[k][c["load"]] += c["weight"] * (b["median_ns"] / 1e3)
    print("\nOur saved per-load times (us) vs SGLang:")
    for L in loads:
        print(f"  {L:10s}: dec={our_load['muse_decode_layer'][L]:8.1f} (sp {sg_load['muse_decode_layer'][L]/our_load['muse_decode_layer'][L]:.3f})"
              f"  sum(bf16+attn+moe)={our_load['muse_bf16_proj'][L]+our_load['muse_attention_decode'][L]+our_load['muse_moe_w4a16'][L]:8.1f}")


if __name__ == "__main__":
    main()
