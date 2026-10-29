"""Simulate exact work decomposition in moe.cu across M=1,8,16,32,128,256.

Models:
1. TINY/TINY2 (M<=8): skinny_ring_ws (256-row/16-row units? RB_A=512, RB_B=256 16-row blocks per expert).
2. MID (M=9..16): -DOKC_TINY=1 -DOKC_SPLIT=1 -> splits each expert with cnt > 8 into ceil(cnt/8) skinny jobs,
   re-streaming all 28.3 MB of that expert's weights for EACH split job!
3. MAIN/WGS (M>=17): -DOKC_WGS=1 -> 256-row units (32 in phase A, 16 in phase B) x 16 stages, stream-K split
   across G=132 CTAs. Counts how many units are whole vs split across CTAs, and how many stages per CTA.
4. Non-split skinny+fat or hybrid alternatives.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import routing_comparison as rc

G = 132  # H200 SMs
MB_EXP = (8192 * 2048 + 8192 * 256 + 4096 * 2048 + 4096 * 256) / 1e6  # 28.311552 MB
WGS_CA = [112, 135, 140, 158, 173, 185, 200, 207]
WGS_CB = [100, 104, 115, 123, 132, 144, 156, 174]


def wgs_cost(pb, cnt):
    ntt = min(8, max(1, (cnt + 7) >> 3))
    return (WGS_CB if pb else WGS_CA)[ntt - 1]


def sim_wgs(counts):
    # Jobs: fat (cnt > 8, chunks of 64) then skinny (0 < cnt <= 8)
    jC = []
    for tot in counts.tolist():
        if tot > 8:
            q = 0
            while q * 64 < tot:
                jC.append(min(64, tot - q * 64))
                q += 1
        elif tot > 0:
            jC.append(tot)
    nj = len(jC)
    if nj == 0:
        return {}
    out = {}
    for pb, name, u_per_job in ((0, "A", 32), (1, "B", 16)):
        sh = 9 - pb
        pre = [0]
        for c in jC:
            pre.append(pre[-1] + (wgs_cost(pb, c) << sh))
        tot = pre[-1]
        def stage_at(x):
            lo, hi = 0, nj
            while hi - lo > 1:
                mid = (lo + hi) >> 1
                if pre[mid] <= x:
                    lo = mid
                else:
                    hi = mid
            cs = wgs_cost(pb, jC[lo])
            return (lo << sh) + (x - pre[lo]) // cs
        whole = split_u = total_seg = 0
        stages_per_cta = []
        for c in range(G):
            g0 = stage_at(tot * c // G)
            g1 = (nj << sh) if c == G - 1 else stage_at(tot * (c + 1) // G)
            stages_per_cta.append(g1 - g0)
            gs = g0
            while gs < g1:
                u = gs >> 4
                s0 = gs & 15
                s1 = min(16, s0 + (g1 - gs))
                total_seg += 1
                if s1 - s0 == 16:
                    whole += 1
                elif s0 == 0:
                    split_u += 1
                gs += s1 - s0
        out[name] = (nj * u_per_job, whole, split_u, total_seg, sum(stages_per_cta)/G, max(stages_per_cta))
    return out


def main():
    shapes = [("decode 1", 1, 1), ("verify 1", 1, 8), ("decode 16", 16, 1),
              ("decode 32", 32, 1), ("verify 16", 16, 8), ("verify 32", 32, 8)]
    for track_name, fn in (("MoE track", lambda b, q, s: rc.moe_track_sample(b * q, s)),
                           ("Decode track", lambda b, q, s: rc.decode_track_sample(b, q, s))):
        print(f"=== {track_name} ===")
        for name, b, q in shapes:
            T = b * q
            dist_l, split_jobs_l, wgsA_split, wgsB_split, wgsA_stg, wgsB_stg = [], [], [], [], [], []
            for s in range(1, 33):
                _, d, c = fn(b, q, 3000 + s)
                dist_l.append(d)
                split_jobs = sum((int(x) + 7) // 8 for x in c if x > 0)
                split_jobs_l.append(split_jobs)
                w = sim_wgs(c)
                if w:
                    wgsA_split.append(w["A"][2] / w["A"][0])
                    wgsB_split.append(w["B"][2] / w["B"][0])
                    wgsA_stg.append(w["A"][4])
                    wgsB_stg.append(w["B"][4])
            md = sum(dist_l)/len(dist_l)
            ms = sum(split_jobs_l)/len(split_jobs_l)
            print(f"{name:10s} (T={T:3d}): distinct={md:5.2f} ({md*MB_EXP/4.8:5.1f}us) | "
                  f"if OKC_SPLIT(<=8): jobs={ms:5.2f} (+{(ms-md)*MB_EXP/4.8:4.1f}us re-stream, {(ms/md-1)*100:4.1f}%) | "
                  f"if WGS: stg/CTA A={sum(wgsA_stg)/len(wgsA_stg):5.1f} (split {sum(wgsA_split)/len(wgsA_split)*100:4.1f}%), "
                  f"B={sum(wgsB_stg)/len(wgsB_stg):5.1f} (split {sum(wgsB_split)/len(wgsB_split)*100:4.1f}%)")


if __name__ == "__main__":
    main()
