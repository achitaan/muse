"""Offline sensitivity model for muse_decode_layer (no GPU, no network).

Inputs are the saved per-shape benchmark medians (SGLang and retained D9,
job 9560c915e85f4f50) and the D9 profile kernel durations.  It answers:
"how much per-shape time must be removed, and where, to reach a target
track speedup?"  Every output is arithmetic on saved numbers, not a forecast
of what an implementation would achieve.
"""
import json, math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROF = ROOT / "results/decoder-segment-repair-20261028/profiles/submission.9560c915e85f4f50.json"
LOADS = ["decode 1", "decode 16", "decode 32", "verify 1", "verify 16", "verify 32"]
TOKENS = {"decode 1": 1, "decode 16": 16, "decode 32": 32, "verify 1": 8, "verify 16": 128, "verify 32": 256}
W = {"sliding": 46, "full": 16}
FLOOR = {("decode 1", "sliding"): 40.0, ("decode 1", "full"): 44.2,
         ("decode 16", "sliding"): 76.7, ("decode 16", "full"): 137.5,
         ("decode 32", "sliding"): 84.5, ("decode 32", "full"): 208.1,
         ("verify 1", "sliding"): 62.3, ("verify 1", "full"): 66.4,
         ("verify 16", "sliding"): 126.6, ("verify 16", "full"): 188.7,
         ("verify 32", "sliding"): 143.8, ("verify 32", "full"): 267.2}


def load_shapes():
    d = json.loads(PROF.read_text())
    shapes = {}
    for b in d["benchmarks"]:
        c = b["case"]
        key = (c["load"], c["layer"])
        sg = b["baseline_ns"] / 1e3
        ours = b["median_ns"] / 1e3
        shapes[key] = {"sglang": sg, "ours": ours, "ratio": b["paired_ratio"]}
    for p in d["profiles"]:
        c = p["case"]
        key = (c["load"], c["layer"])
        kern = {}
        for line in p["report"].splitlines()[1:]:
            if line.startswith("total"):
                continue
            parts = line.rsplit(None, 4)
            if len(parts) == 5 and parts[2] == "µs" or (len(parts) == 5 and parts[2].endswith("s")):
                kern[parts[0].strip()] = float(parts[1])
        shapes[key]["kernels"] = kern
    return shapes


def score(times, shapes):
    logs = []
    for L in LOADS:
        sg = sum(W[l] * shapes[(L, l)]["sglang"] for l in W)
        us = sum(W[l] * times[(L, l)] for l in W)
        logs.append(math.log(sg / us))
    return math.exp(sum(logs) / len(logs)), [math.exp(x) for x in logs]


def main():
    s = load_shapes()
    base = {k: v["ours"] for k, v in s.items()}
    sp, per = score(base, s)
    print(f"reconstructed speedup {sp:.4f}; per load " + ", ".join(f"{L}={x:.3f}" for L, x in zip(LOADS, per)))
    print("\nshape                      sglang    ours   floor  ours/floor  attn_kernel moe_kernel")
    for k in sorted(s, key=lambda k: (LOADS.index(k[0]), k[1])):
        kern = s[k].get("kernels", {})
        attn = next((v for n, v in kern.items() if "attn_kernel" in n), float("nan"))
        moe = next((v for n, v in kern.items() if "moe_fused" in n), float("nan"))
        print(f"{k[0]:10s} {k[1]:8s} {s[k]['sglang']:8.1f} {s[k]['ours']:7.1f} {FLOOR[k]:7.1f}"
              f" {s[k]['ours']/FLOOR[k]:8.2f}    {attn:8.1f} {moe:8.1f}")

    targets = {"rank~150 (2.091)": 2.0908, "rank~100 (2.123)": 2.1225,
               "rank~50 (2.184)": 2.1836, "rank~25 (2.235)": 2.2348}
    print("\nA. uniform absolute saving per shape (all 12 shapes) needed:")
    for name, t in targets.items():
        lo, hi = 0.0, 80.0
        for _ in range(60):
            mid = (lo + hi) / 2
            v, _ = score({k: base[k] - mid for k in base}, s)
            lo, hi = (mid, hi) if v < t else (lo, mid)
        print(f"   {name}: {hi:5.2f} us per shape")
    print("\nB. uniform fractional saving per shape needed:")
    for name, t in targets.items():
        f = 1 - sp / t
        print(f"   {name}: {f*100:5.2f}% of every shape")

    print("\nC. single-kernel-class what-ifs (time removed = fraction of that kernel's profile time;"
          " assumes it is on the critical path, an upper bound):")
    def whatif(pred, frac):
        t = {}
        for k in base:
            kern = s[k].get("kernels", {})
            rem = sum(v for n, v in kern.items() if pred(n, k))
            t[k] = base[k] - frac * rem
        return score(t, s)[0]
    cases = {
        "MoE kernel 15% faster everywhere": (lambda n, k: "moe_fused" in n, 0.15),
        "MoE kernel 25% faster everywhere": (lambda n, k: "moe_fused" in n, 0.25),
        "full-layer attention 20% faster": (lambda n, k: "attn_kernel" in n and k[1] == "full", 0.20),
        "full-layer attention 30% faster": (lambda n, k: "attn_kernel" in n and k[1] == "full", 0.30),
        "all norm/boundary kernels removed": (lambda n, k: ("norm_residual" in n or "_sfn" in n or "_pn_inputs" in n), 1.0),
        "small-token route_fp32 removed": (lambda n, k: "route_fp32" in n, 1.0),
    }
    for name, (pred, frac) in cases.items():
        print(f"   {name:40s} -> {whatif(pred, frac):.4f}")
    combo = lambda n, k: ("moe_fused" in n and 0.15) or ("attn_kernel" in n and k[1] == "full" and 0.2) or 0
    t = {k: base[k] - sum(v * combo(n, k) for n, v in s[k].get("kernels", {}).items()) for k in base}
    print(f"   {'MoE -15% + full attention -20%':40s} -> {score(t, s)[0]:.4f}")


if __name__ == "__main__":
    main()
