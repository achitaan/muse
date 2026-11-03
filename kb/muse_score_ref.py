#!/usr/bin/env python3
"""Offline Muse Decode score replay, matched-control baseline, and conditional scenarios.

python muse_score.py
python muse_score.py --save-us d16S=10 --save-us d16F=8
python muse_score.py --job kb/runs/keell_dec.cde2084a.json
Positive microseconds are savings on the whole measured layer; negative values are regressions.
"""
# achitaan: user-supplied version (Nov 3), kept verbatim as kb/muse_score_ref.py next to kb/muse_score.py.
import argparse
import json
import math
import statistics
from pathlib import Path

LOADS = ("d1", "d16", "d32", "v1", "v16", "v32")
# f2ab6d4c check times from the repo's JOEY_PLAN.md, in microseconds, sliding/full.
PROFILE = dict(zip(LOADS, [(65.5, 74.1), (126.8, 202.0), (142.6, 305.3),
                           (101.7, 110.2), (238.5, 329.5), (314.2, 473.4)]))
SCORED_ANCHOR = 2.2347  # 1ca0668a, same effective scored path as kb v1.
# Seven distinct checks reported as default/no-adoption in main:TRIED.md.
# This is a descriptive reference cohort, not a random sample or a prediction interval.
CONTROLS = {"afe3c0e2": 2.2318, "7fd03cfa": 2.2607, "4dcc1d37": 2.2424,
            "781d89f7": 2.2412, "b3e93db2": 2.2433, "f2ab6d4c": 2.2445,
            "7bae45c5": 2.2489}


def positive(x):
    x = float(x)
    if not math.isfinite(x) or x <= 0:
        raise ValueError("Expected a finite, positive number")
    return x


def job_profile(path):
    job = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if "benchmarks" in job:
        result = job["summary"]
        board = job.get("problem")
        records = job["benchmarks"]
    else:
        result = job["result"]
        board = job.get("board")
        records = result["records"]
    if board != "muse_decode_layer" or result.get("passed") is not True or result.get("flags"):
        raise ValueError("Expected a passing, unflagged muse_decode_layer result")
    slots = {}
    for rec in records:
        if rec.get("event") != "benchmark":
            continue
        case = rec["case"]
        kind, batch = case["load"].split()
        load = {"decode": "d", "verify": "v"}[kind] + batch
        layer = {"sliding": "S", "full": "F"}[case["layer"]]
        key = load + layer
        if key in slots or load not in LOADS or rec.get("flags") or rec.get("status") != "pass":
            raise ValueError(f"Unexpected, repeated, or invalid case {key}")
        slots[key] = (positive(case["weight"]), positive(rec["baseline_ns"]) / 1000,
                      positive(rec["median_ns"]) / 1000)
    if set(slots) != {load + layer for load in LOADS for layer in "SF"}:
        raise ValueError("Expected exactly 12 scored cases")
    ratios = []
    for load in LOADS:
        cells = [slots[load + layer] for layer in "SF"]
        ratios.append(sum(w * b for w, b, _ in cells) / sum(w * t for w, _, t in cells))
    score = statistics.geometric_mean(ratios)
    if not math.isclose(score, positive(result["speedup"]), rel_tol=1e-6):
        raise ValueError("Score reconstruction disagrees with the official record")
    return slots, score


def scenario(slots, anchor, savings):
    ratios = []
    for load in LOADS:
        before = after = 0.0
        for layer in "SF":
            key = load + layer
            weight, _, latency = slots[key]
            proposed = positive(latency - savings.get(key, 0))
            before += weight * latency
            after += weight * proposed
        ratios.append(before / after)
    return anchor * statistics.geometric_mean(ratios)


# Numeric revision, job ID, decode-16 load ratio, verify-1 load ratio, overall ratio.
# Compact observations from cf36b14/experiments/top50/profiles/decoder_*.json.
# Eligible: 12 passing benchmark cases, passing job, archived score reconstructed.
# The unnumbered diagnostic is excluded from the revision split. Multiple jobs from
# one revision count as one family for fitting and for reporting test MAE.
HISTORY = [
    (1, "0ca7186139354bd4", 1.9646755849210908, 2.2928571722383073, 2.0975444321115506),
    (1, "7abd8d0d9da741a5", 1.9410373009944488, 2.221350045409857, 2.04341533526596),
    (2, "ac41a480936c4d0e", 1.9492306040321534, 2.2520212872693324, 2.081340393096328),
    (3, "f82e5ab86c5f4ca2", 1.9339079327998627, 2.242253535551297, 2.0502798453444746),
    (4, "d543da405e89487d", 1.9533896230391206, 2.2778832765755395, 2.089144350847265),
    (5, "6553b299527d4fa6", 1.9712141131924836, 2.2474207188279522, 2.0960212665422997),
    (5, "831de9a6fc164b0a", 1.9390687485309814, 2.26088375767522, 2.0900145652392057),
    (6, "0aa30c9234014956", 1.945831022529598, 2.2685765555940773, 2.0783334239549713),
    (6, "689fd93e189943a0", 1.9725471423694687, 2.2676588560336346, 2.068206362608522),
    (7, "97c2ab828d524da9", 1.9475339235932672, 2.2494335120928963, 2.0806303255829772),
    (7, "b2574b499e6f495a", 1.9418759207001715, 2.3088958566378843, 2.0666634902742227),
    (7, "db30df751c174999", 1.9742174056620374, 2.261464425941242, 2.076626141491534),
    (8, "2ef9150c869e45fa", 1.9462938239011374, 2.2488680776882632, 2.0671259197415957),
    (8, "e057dc9de2af4b82", 1.9638806203172452, 2.2607597659462852, 2.077630625786861),
    (9, "c5f34d4fb33145aa", 1.9583859374633583, 2.294597534210187, 2.094195274136709),
    (10, "e1af1dd068e74521", 1.9783236655542118, 2.2548358035888625, 2.0969964500086906),
    (10, "f9845bc9c0eb4269", 1.9502804578357944, 2.2974465634412313, 2.104304047646603),
    (12, "c1d5a6f15e514101", 1.950674190446584, 2.234851280794905, 2.0711166266081418),
    (14, "182d084ec48e41da", 1.9632309652198556, 2.2603535470215856, 2.0656145447548266),
    (15, "f35cb9c29e8248ab", 1.985991947356785, 2.252860546934253, 2.0926471395450843),
    (16, "d23cc6a256834a41", 1.965564136411738, 2.2635792491720625, 2.081399376674138),
    (17, "848a03acfb1a4843", 1.953987237074303, 2.2899845631633524, 2.07171232136679),
    (18, "482d0e7cdb404ad9", 1.9297614198019928, 2.2519677671955707, 2.074128275530744),
    (19, "9b254bbb1366412e", 1.925929578238112, 2.2627639335194822, 2.0717661257639675),
    (20, "0898ae00e7014b89", 1.9538766257352056, 2.260010258911097, 2.0709154193847827),
    (21, "f67edab834e44935", 1.9318072329673577, 2.2728249534417424, 2.0792114633472862),
    (22, "15e19cf804234910", 1.9405228451158245, 2.260462362984805, 2.06977722712637),
    (23, "2cbdfa2acbb143dc", 1.9382060328042587, 2.242124987816767, 2.0667374883730076),
    (24, "a6aedcc2df3942ec", 1.9659809179737289, 2.2515489600230394, 2.060708514584862),
    (25, "3572476680f4460a", 1.953814399166341, 2.259963292584149, 2.0876593126761667),
    (25, "a78368454a1c4c17", 1.9975182772926168, 2.29625487165652, 2.110676729784457),
    (28, "17ef70614da5496e", 1.9479368711263498, 2.291758388925573, 2.0834289910927004),
    (31, "2014d84eb9184999", 1.9415279982782365, 2.246588640475506, 2.0786991472958074),
    (32, "69d26d1c04b446bc", 1.986748825237964, 2.2546968346602996, 2.088328730921988),
    (32, "ba1e91bbcc2f49fc", 2.00974130688944, 2.2996910576923217, 2.083984526626972),
    (33, "d2d2daa4004c4cb3", 1.9732962997136152, 2.2596201807414347, 2.0667010323938086),
    (34, "a40cbaf0843e4316", 1.9869067970136525, 2.2700722567809066, 2.086728941711152),
    (34, "a7efb0cda8eb4b32", 1.9921600855614248, 2.273922054093795, 2.087955703267826),
    (35, "412977a6ce9545d4", 1.9736692796012798, 2.2356104764925036, 2.086518514503325),
    (35, "cba4891a02094df0", 1.9671098981041015, 2.2743238249410327, 2.0562126149700504),
    (36, "22240ad3e3f04f41", 1.9791970068329023, 2.262233895558, 2.117813483956192),
    (36, "9dabcd9a13554bad", 1.96987549790177, 2.278684606696943, 2.1056283229771564),
    (37, "3056edef63cd4418", 1.9820503775611567, 2.247477546821538, 2.081555455333855),
]
# Later independent archive checks on two different code bases (alygg and Keell).
TRANSFER = [
    (100, "3bd3d93a06fc418b", 2.053597821105015, 2.4723266294504045, 2.208449518952965),
    (101, "cde2084a8e9748c7", 2.119782009151599, 2.5231771991180243, 2.258780069792867),
]


def fit_partial_model():
    """Predict mean log speedup on four hidden loads from two observed load ratios."""
    by_revision = {}
    for rev, _, d16, v1, total in HISTORY:
        if rev < 25:
            x1, x2 = math.log(d16), math.log(v1)
            hidden = (6 * math.log(total) - x1 - x2) / 4
            by_revision.setdefault(rev, []).append((x1, x2, hidden))
    samples = [tuple(map(statistics.mean, zip(*group))) for group in by_revision.values()]
    center = tuple(map(statistics.mean, zip(*samples)))
    sd = tuple(statistics.pstdev(row[i] for row in samples) for i in (0, 1))
    standardized = [((a-center[0])/sd[0], (b-center[1])/sd[1], y-center[2])
                    for a, b, y in samples]
    cross = statistics.mean(a*b for a, b, y in standardized)
    xy1 = statistics.mean(a*y for a, b, y in standardized)
    xy2 = statistics.mean(b*y for a, b, y in standardized)
    # Ridge penalty of one on standardized inputs and mean squared loss.
    det = 4 - cross**2
    return center, sd, ((2*xy1-cross*xy2)/det, (2*xy2-cross*xy1)/det)


def partial_score(d16, v1, fit, *, use_slopes=True):
    center, sd, slopes = fit
    x = (math.log(positive(d16)), math.log(positive(v1)))
    hidden = center[2]
    if use_slopes:
        hidden += sum(b * (v-m)/s for b, v, m, s in zip(slopes, x, center[:2], sd))
    return math.exp((sum(x) + 4*hidden)/6)


def audit_partial_model(fit):
    held = [row for row in HISTORY if row[0] >= 25]
    revisions = sorted({row[0] for row in held})
    def mae(use_slopes):
        errors = [statistics.mean(abs(partial_score(d, v, fit, use_slopes=use_slopes) - s)
                                  for r, _, d, v, s in held if r == rev)
                  for rev in revisions]
        return statistics.mean(errors)
    print("Historical trial: fit P1–P24 (29 jobs in 22 revision groups), "
          "hold out P25–P37 (14 jobs in 9 groups).")
    print(f"MAE, equal weight per held-out revision: ridge {mae(True):.5f}x, "
          f"two observed loads + constant for the other four {mae(False):.5f}x.")
    for _, job, d16, v1, score in TRANSFER:
        estimate = partial_score(d16, v1, fit)
        print(f"Different code base {job}: predicted {estimate:.5f}x, observed {score:.5f}x, "
              f"error {estimate-score:+.5f}x.")
    print("Inputs are two complete normalized load measurements from each held-out job. "
          "Live proxy-to-judge transfer and pre-benchmark source-only ranking remain unvalidated.")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--job", help="official archived Decode job JSON for exact replay")
    p.add_argument("--anchor", type=positive, help="explicitly re-anchor the scenario speedup")
    p.add_argument("--save-us", metavar="CASE=US", action="append", default=[],
                   help="signed whole-layer time saved, e.g. d16S=10 or v1F=-2")
    p.add_argument("--backtest", action="store_true", help="run the old-lineage grouped holdout and cross-lineage audit")
    p.add_argument("--legacy-probe", nargs=2, type=positive, metavar=("D16", "V1"),
                   help="historical two-probe estimate using complete decode-16 and verify-1 load speedups")
    args = p.parse_args()
    if args.backtest or args.legacy_probe:
        if args.job or args.save_us or args.anchor:
            p.error("Use the backtest or two-load estimator separately from a timing scenario")
        fit = fit_partial_model()
        if args.backtest:
            audit_partial_model(fit)
        if args.legacy_probe:
            print(f"Historical-family estimate: {partial_score(*args.legacy_probe, fit):.5f}x")
            print("Research only. This fit underpredicted newer kernel families in the transfer audit.")
        return
    try:
        if args.job:
            slots, anchor = job_profile(args.job)
            print(f"Reconstructed archived job: {anchor:.8f}x")
        else:
            slots = {load + layer: (weight, None, PROFILE[load][i])
                     for load in LOADS for i, (layer, weight) in enumerate([("S", 46), ("F", 16)])}
            anchor = SCORED_ANCHOR
            print("Default: f2ab6d4c timing profile re-anchored to the 2.2347x scored path.")
        savings = {}
        for spec in args.save_us:
            key, raw = spec.split("=", 1)
            value = float(raw)
            if key not in slots or key in savings or not math.isfinite(value):
                raise ValueError(f"Unknown or repeated case, or invalid delta: {spec}")
            savings[key] = value
        anchor = args.anchor if args.anchor is not None else anchor
        score = scenario(slots, anchor, savings)
    except (KeyError, ValueError, TypeError, OSError) as exc:
        p.error(str(exc))
    print(f"Conditional scenario: {score:.5f}x ({100 * (score / anchor - 1):+.2f}% score change).")
    print("Other shapes held fixed. An isolated kernel saving must be converted to exposed layer time first.")
    vals = list(CONTROLS.values())
    print(f"Historical unchanged-path Decode check baseline (n={len(vals)}): "
          f"median {statistics.median(vals):.4f}x, observed range {min(vals):.4f}–{max(vals):.4f}x.")
    print("This reference cohort applies to kb v1's default path only; it is not a confidence interval.")


if __name__ == "__main__":
    main()
