"""CPU-only checks for WQ: NVRTC sm_90a compile of all four MoE builds, and a model of the rounded stream-K ranges."""
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OPUS = HERE.parents[2] / 'opus-fresh' / 'experiments' / 'opus_fresh'
sys.path.insert(0, str(OPUS))
import nvrtc_screen_alternatives as nv  # noqa: E402

FWD_OLD = "__device__ __forceinline__ int wgs_cta_of(const Smem& S, int pb, int s, int G);\n"
FWD_NEW = FWD_OLD
ANCHOR = "template <bool PB>\n__device__ __forceinline__ void wgs_phase"
BUILDS = {
    "MAIN": ["-DOKC_NW=16", "-DOKC_FAST1=0", "-DOKC_EPI4=0", "-DOKC_WGS=1", "-DOKC_PRMTSC=1"],
    "TINY": ["-DOKC_NW=8", "-DOKC_TINY=1", "-DOKC_EPI4=0", "-DOKC_FAST1=2", "-DOKC_L2H=0", "-DOKC_NACC=4",
             "-DOKC_UWARP=0", "-DOKC_WS=1", "-DOKC_WSB=1", "-DOKC_WSBL=2", "-DOKC_WSBLK=2", "-DOKC_FINPF=1",
             "-DOKC_PRMTSC=1"],
    "MID": ["-DOKC_NW=8", "-DOKC_TINY=1", "-DOKC_SPLIT=1", "-DOKC_CPF=0", "-DOKC_EPI4=0", "-DOKC_FAST1=2",
            "-DOKC_NACC=4", "-DOKC_UWARP=0", "-DOKC_WS=1", "-DOKC_WSB=1", "-DOKC_WSBL=2", "-DOKC_WSBLK=2",
            "-DOKC_FINPF=1", "-DOKC_PRMTSC=1"],
}


def compile_all(nvrtc, cuda, fwd):
    src = cuda.split("#include <torch/extension.h>", 1)[0].replace(ANCHOR, fwd + ANCHOR, 1).encode()
    out = {}
    for name, defs in BUILDS.items():
        r = nvrtc.compile(src, ["&okc::moe_fused<false>", "&okc::moe_fused<true>"],
                          defs + ["-DOKC_DQ7=1", "-DOKC_NS=3", "-DOKC_NSF=3"], tags=True, lineinfo=False)
        out[name] = {"rc": r["rc"], "kernels": nv.parse_ptxas(r["log"]) if r["rc"] == 0 else r["log"][-1500:]}
    return out


# ---- model of wgs_prefix / wgs_stage_at / wgs_bound / wgs_cta_of
CA = [112, 135, 140, 158, 173, 185, 200, 207]
CB = [100, 104, 115, 123, 132, 144, 156, 174]


def cost(pb, c):
    return (CB if pb else CA)[((c + 7) >> 3) - 1]


def model(jc, G, pb, q):
    sh = 9 - pb
    pre = [0]
    for c in jc:
        pre.append(pre[-1] + (cost(pb, c) << sh))
    tot, nj, end = pre[-1], len(jc), len(jc) << sh
    assert tot * G < 2 ** 32

    def stage_at(x):
        lo, hi = 0, nj
        while hi - lo > 1:
            mid = (lo + hi) >> 1
            lo, hi = (mid, hi) if pre[mid] <= x else (lo, mid)
        return (lo << sh) + (x - pre[lo]) // cost(pb, jc[lo])

    def bound(c):
        if c >= G:
            return end
        g = stage_at(tot * c // G)
        return (g + (q >> 1)) // q * q

    def owner(s):
        lo, hi = 0, G
        while hi - lo > 1:
            mid = (lo + hi) >> 1
            lo, hi = (mid, hi) if bound(mid) <= s else (lo, mid)
        return lo

    b = [bound(c) for c in range(G + 1)]
    assert b[0] == 0 and b[G] == end and all(b[i] <= b[i + 1] for i in range(G))
    assert all(x % q == 0 for x in b)
    split_units, firsts = 0, [0] * G
    for c in range(G):
        g0, g1 = b[c], b[c + 1]
        for s in range(g0, g1):
            assert owner(s) == c
        gs = g0
        while gs < g1:
            s0, s1 = gs & 15, min(16, (gs & 15) + g1 - gs)
            if s1 - s0 < 16:
                if s0 == 0:
                    firsts[c] += 1
                    split_units += 1
                else:
                    assert owner((gs >> 4) << 4) != c
            gs += s1 - s0
    assert max(firsts) <= 1  # fixup slot uniqueness: one split unit's first stage per CTA
    units = end >> 4
    per_cta = [b[c + 1] - b[c] for c in range(G)]
    return {"split_frac": split_units / units, "max_stages": max(per_cta), "mean_stages": sum(per_cta) / G}


def main():
    cu_new = (HERE / 'moe_wq.cu').read_text(encoding='utf-8')
    cu_old = (HERE / 'moe_parent.cu').read_text(encoding='utf-8')
    nvrtc = nv.NVRTC()
    res = {"parent": compile_all(nvrtc, cu_old, FWD_OLD), "wq": compile_all(nvrtc, cu_new, FWD_NEW)}
    rnd = random.Random(11)
    stats = {}
    for trial in range(300):
        nj = rnd.choice([4, 6, 7, 8, 12, 17, 24, 40])
        jc = [rnd.randint(1, 64) for _ in range(nj)]
        G = rnd.choice([132, 120, 100, 66])
        for pb in (0, 1):
            for q in (1, 2, 4, 8, 16):
                m = model(jc, G, pb, q)
                if nj == 7 and G == 132:
                    stats.setdefault(f"nj7 pb{pb} q{q}", []).append((round(m["split_frac"], 3), m["max_stages"]))
    res["model"] = "300 random job sets x G {132,120,100,66} x q {1,2,4,8,16}: partition, owner lookup, slot uniqueness OK"
    res["nj7_examples"] = {k: v[:3] for k, v in stats.items()}
    (HERE / 'check_wq_cpu.json').write_text(json.dumps(res, indent=1), encoding='utf-8')
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
