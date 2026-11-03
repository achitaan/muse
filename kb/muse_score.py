#!/usr/bin/env python3
# achitaan: Muse Open Kernel Challenge score calculator, backtester and what-if tool.
# Score = geometric mean over loads of  sum(weight * baseline_ns) / sum(weight * median_ns)
# using the judge's adjusted per-case fields in `okc status --json` records (event == "benchmark").
#
#   python muse_score.py --backtest [DIR_OR_JSON ...]            replay saved jobs, compare with the judge's speedup
#   python muse_score.py --job f2ab6d4c --save-us d16S=10 --save-us d16F=8
#   python muse_score.py --job f2ab6d4c --scale d16S=0.92        multiply that shape's time
#   python muse_score.py --job f2ab6d4c --table                  per-shape table
# Shape keys: d1S d1F d16S d16F d32S d32F v1S v1F v16S v16F v32S v32F (load initial + batch + S/F) for Decode;
# for other boards the key is "<load>|<layer or ''>" as printed by --table.
import argparse, glob, json, math, os, sys

DEFAULT_DIRS = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs"),
                r"C:\Users\Achita\Documents\ChatGPT\Muse 2\.slim\worktrees\kbase\kb\runs"]


def load_job(path):
    with open(path, encoding="utf-8-sig") as f:
        j = json.load(f)
    res = j.get("result") or {}
    recs = [r for r in res.get("records", []) if r.get("event") == "benchmark"]
    return j, res, recs


def key_of(case):
    load = case.get("load", "")
    layer = case.get("layer", "")
    parts = load.split()
    if len(parts) == 2 and layer in ("sliding", "full"):
        return f"{parts[0][0]}{parts[1]}{'S' if layer == 'sliding' else 'F'}"
    return f"{load}|{layer}"


def score(recs, save_us=None, scale=None):
    save_us = save_us or {}
    scale = scale or {}
    loads = {}
    for r in recs:
        c = r["case"]
        k = key_of(c)
        med = float(r["median_ns"]) * scale.get(k, 1.0) - 1000.0 * save_us.get(k, 0.0)
        if med <= 0:
            raise ValueError(f"non-positive time for {k}")
        w = float(c.get("weight", 1))
        b, m = loads.setdefault(c["load"], [0.0, 0.0])
        loads[c["load"]] = [b + w * float(r["baseline_ns"]), m + w * med]
    ratios = {L: b / m for L, (b, m) in loads.items()}
    return math.exp(sum(math.log(x) for x in ratios.values()) / len(ratios)), ratios


def find_job(token, dirs):
    for d in dirs:
        for p in glob.glob(os.path.join(d, "*.json")):
            if token in os.path.basename(p):
                return p
    if os.path.isfile(token):
        return token
    sys.exit(f"job {token} not found in {dirs}")


def parse_kv(items, cast=float):
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        out[k] = cast(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backtest", nargs="*", metavar="PATH")
    ap.add_argument("--job", default="1ca0668a")
    ap.add_argument("--save-us", action="append", metavar="SHAPE=US")
    ap.add_argument("--scale", action="append", metavar="SHAPE=FACTOR")
    ap.add_argument("--table", action="store_true")
    a = ap.parse_args()

    if a.backtest is not None:
        paths = []
        for p in (a.backtest or DEFAULT_DIRS):
            paths += glob.glob(os.path.join(p, "*.json")) if os.path.isdir(p) else [p]
        n = bad = 0
        worst = 0.0
        for p in sorted(paths):
            try:
                j, res, recs = load_job(p)
            except Exception:
                continue
            if not recs or res.get("speedup") is None:
                continue
            s, _ = score(recs)
            err = s - float(res["speedup"])
            n += 1
            worst = max(worst, abs(err))
            if abs(err) > 5e-4:
                bad += 1
                print(f"MISMATCH {os.path.basename(p)} calc {s:.4f} judge {float(res['speedup']):.4f}")
        print(f"backtest: {n} jobs, {bad} mismatches, max |err| {worst:.2e}")
        return

    p = find_job(a.job, DEFAULT_DIRS)
    j, res, recs = load_job(p)
    base, base_r = score(recs)
    print(f"job {os.path.basename(p)} judge {res.get('speedup')} calc {base:.4f}")
    if a.table:
        for r in recs:
            c = r["case"]
            print(f"  {key_of(c):>6} w{c.get('weight')} base {r['baseline_ns']/1e3:8.2f} us  ours {r['median_ns']/1e3:8.2f} us  x{r['baseline_ns']/r['median_ns']:.3f}")
        for L, x in base_r.items():
            print(f"  load {L:>10}: {x:.4f}")
    sv, sc = parse_kv(a.save_us), parse_kv(a.scale)
    if sv or sc:
        new, new_r = score(recs, sv, sc)
        print(f"what-if {sv or ''} {sc or ''}: {base:.4f} -> {new:.4f} ({(new / base - 1) * 100:+.2f}%)")
        for L in base_r:
            if abs(new_r[L] / base_r[L] - 1) > 1e-9:
                print(f"  load {L:>10}: {base_r[L]:.4f} -> {new_r[L]:.4f}")


if __name__ == "__main__":
    main()
