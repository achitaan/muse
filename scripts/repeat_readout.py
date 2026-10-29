"""Apply the frozen repeat-run rules (results/repeat-plan-20261029.json) to every finished run."""
import json
import math
from pathlib import Path

DF = Path(__file__).resolve().parents[1]
OT = DF.parent / "other-tracks"


def load(path, key):
    f = Path(path) / "result.json"
    if not f.exists() or '"records"' not in f.read_text(encoding="utf-8"):
        return None
    r = json.load(open(f, encoding="utf-8"))
    r = r.get("result", r)
    out = {}
    for x in r["records"]:
        if x.get("event") == "benchmark" and isinstance(x.get("case"), dict):
            out[key(x["case"])] = x["median_ns"] / 1000.0
    return out


def pairwise(cands, bases, targets):
    """Mean over (candidate, base) pairs of base/candidate on each target, normalized by the
    geometric mean of base/candidate over all non-target shapes present in both runs."""
    res = {t: [] for t in targets}
    for c in cands.values():
        for b in bases.values():
            ctrl = [k for k in c if k in b and k not in targets]
            gm = math.exp(sum(math.log(b[k] / c[k]) for k in ctrl) / len(ctrl))
            for t in targets:
                res[t].append((b[t] / c[t]) / gm)
    return {t: (sum(v) / len(v), min(v), len(v)) for t, v in res.items()}


pkey = lambda c: (c["load"], c["layer"], c["ctx"], c["batch"])
p7 = {n: load(OT / "results" / d, pkey) for n, d in [("P7", "prefill-kv-gather-triton-p7-20261029"), ("p71", "repeat-p71-20261029"), ("p72", "repeat-p72-20261029"), ("p73", "repeat-p73-20261029")]}
pp = {n: load(OT / "results" / d, pkey) for n, d in [("P1", "prefill-fa-select-check-20261029"), ("P2", "prefill-moe-sched-check-20261029"), ("P3", "prefill-gemm-layout-check-20261029"), ("pctl1", "repeat-pctl1-20261029"), ("pctl2", "repeat-pctl2-20261029"), ("pctl3", "repeat-pctl3-20261029")]}
p7 = {k: v for k, v in p7.items() if v}
pp = {k: v for k, v in pp.items() if v}
sel = [("first turn", "full", 0, 1), ("first turn", "full", 32768, 1), ("first turn", "full", 65536, 1), ("turn 4", "full", 80000, 4)]
print("PREFILL runs: P7-path", sorted(p7), "parent-path", sorted(pp))
pr = pairwise(p7, pp, sel)
for t, (m, lo, n) in pr.items():
    print(f"  {t}: mean {m:.4f} min {lo:.4f} pairs {n}")
ok = all(m >= 1.02 for (m, _, _) in pr.values()) and all(m >= 1.00 for (m, _, _) in pr.values())
print("  PREFILL RULE (each selected shape mean >= 1.02, none < 1.00):", "MET" if ok else "NOT MET")

akey = lambda c: (c["load"], c["layer"])
a8 = {n: load(DF / "results" / d, akey) for n, d in [("A3", "attention-split-a3-20261029"), ("A5", "attention-autotune-a5-20261029"), ("a3r1", "repeat-a3r1-20261029"), ("a3r2", "repeat-a3r2-20261029")]}
a16 = {n: load(DF / "results" / d, akey) for n, d in [("A6", "attention-control-a6-20261029"), ("actl1", "repeat-actl1-20261029"), ("actl2", "repeat-actl2-20261029")]}
a8 = {k: v for k, v in a8.items() if v}
a16 = {k: v for k, v in a16.items() if v}
t = ("decode 16", "sliding")
ar = pairwise(a8, a16, [t])[t]
print("ATTENTION runs: 8-split", sorted(a8), "16-split", sorted(a16))
print(f"  decode16 sliding 16/8 normalized: mean {ar[0]:.4f} min {ar[1]:.4f} pairs {ar[2]}")
print("  ATTENTION RULE (mean >= 1.03):", "MET" if ar[0] >= 1.03 else "NOT MET")
