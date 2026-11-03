#!/usr/bin/env python3
"""predict.py - honest Decode score predictor + CHASE ranking (offline, stdlib only).

Reuses kb/muse_score.py for the exact judge-score replay and what-ifs.

  python predict.py backtest          replay every Decode job (new + old archive format), reject incomplete, write rows
  python predict.py noise             same-path repeat noise (per shape, per family, score)
  python predict.py transfer          pre->layer, iso->layer, pre->iso transfer: zero / full / shrunk, held out by revision group
  python predict.py gate              adopted in-prepare gate gain -> judge delta (old archive)
  python predict.py overlap           profile-sum vs judged time per shape (critical-path overlap factor)
  python predict.py need              what 2.37 needs, from muse_score what-ifs
  python predict.py chase             ranked experiments (data/candidates.json)
  python predict.py all [--out F]     everything above (this is what BACKTEST.txt contains)
  python predict.py refresh-archive   rebuild data/old_archive_rows.json from `git show cf36b14:...`

Conventions: gains are in us, POSITIVE = faster. Scores are the judge's 6-load geometric mean replayed by muse_score.
"""
import argparse, csv, glob, json, math, os, random, re, statistics as st, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
KB = os.path.dirname(HERE)
sys.path.insert(0, KB)
import muse_score as ms  # noqa: E402

DATA = os.path.join(HERE, "data")
ARCHIVE_COMMIT = "cf36b14"
ARCHIVE_CACHE = os.path.join(DATA, "old_archive_rows.json")
RUN_DIRS = list(ms.DEFAULT_DIRS)
SHAPES = ["d1S", "d1F", "d16S", "d16F", "d32S", "d32F", "v1S", "v1F", "v16S", "v16F", "v32S", "v32F"]
ANCHOR = "1ca0668a"  # retained exact scored reference 2.2347
PROFILE_TXT = os.path.join(KB, "ref", "decode_curp.f2ab6d4c6ba74769.txt")
TOL = 5e-4
OUT = []


def say(*a):
    s = " ".join(str(x) for x in a)
    OUT.append(s)
    print(s)


def jload(p):
    with open(p, encoding="utf-8-sig") as f:
        return json.load(f)


# ----------------------------------------------------------------------------- loading
def norm_case(c):
    return ms.key_of(c)


def families(report):
    """Parse a judge per-kernel profile report into {family: us} (+ total)."""
    fam = {}
    for line in (report or "").splitlines():
        m = re.match(r"^(?P<n>.+?)\s{2,}(?P<t>[\d.]+)\s*\S{0,3}s\s+(?P<sh>[\d.]+)%\s+(?P<c>\d+)\s*$", line.rstrip())
        if not m or m.group("n").strip() == "total":
            continue
        n, t = m.group("n"), float(m.group("t"))
        if "moe_fused" in n:
            f = "moe"
        elif "attn_kernel" in n:
            f = "attention"
        elif any(k in n for k in ("sk2_kernel", "rq<", "nvjet", "v18", "gemv")):
            f = "dense"
        elif any(k in n for k in ("glue_cluster", "_graft_row_boundary", "_sl_cluster_finish", "_sl_tile_route", "_graft_route", "_gle_route")):
            f = "glue_route"
        else:
            f = "other"
        fam[f] = fam.get(f, 0.0) + t
    return fam


def norm_new(path):
    j = jload(path)
    res = j.get("result") or {}
    recs = [r for r in res.get("records", []) if r.get("event") == "benchmark"]
    profs = {r["case"].get("load", "") + "|" + r["case"].get("layer", "") + "|" + str(r["case"].get("batch")): r.get("report", "")
             for r in res.get("records", []) if r.get("event") == "profile"}
    cases = []
    for r in recs:
        c = r["case"]
        k = norm_case(c)
        pk = c.get("load", "") + "|" + c.get("layer", "") + "|" + str(c.get("batch"))
        cases.append(dict(key=k, case=c, median_ns=float(r["median_ns"]), baseline_ns=float(r["baseline_ns"]),
                          sol_ns=r.get("speed_of_light_ns"), report=profs.get(pk, "")))
    ev = res.get("evaluator") or {}
    return dict(id=j.get("id", "")[:8], board=j.get("board"), mode=j.get("mode"), created=j.get("created_at"),
                chash=(j.get("content_hash") or "")[:8], ehash=(ev.get("code_sha256") or "")[:8], fmt="new",
                speedup=res.get("speedup"), passed=res.get("passed"), status=res.get("status"), cases=cases,
                src=os.path.basename(path), recs=[{"case": x["case"], "median_ns": x["median_ns"], "baseline_ns": x["baseline_ns"]} for x in recs])


def norm_old(path_or_obj, name=None):
    j = path_or_obj if isinstance(path_or_obj, dict) else jload(path_or_obj)
    s = j.get("summary") or {}
    cases = []
    profs = {}
    for p in j.get("profiles", []) or []:
        c = p.get("case", {})
        profs[c.get("load", "") + "|" + c.get("layer", "") + "|" + str(c.get("batch"))] = p.get("report", "")
    for b in j.get("benchmarks", []) or []:
        c = b["case"]
        pk = c.get("load", "") + "|" + c.get("layer", "") + "|" + str(c.get("batch"))
        cases.append(dict(key=norm_case(c), case=c, median_ns=float(b["median_ns"]), baseline_ns=float(b["baseline_ns"]),
                          sol_ns=b.get("speed_of_light_ns"), report=profs.get(pk, "")))
    return dict(id=j.get("job_id", "")[:8], board=j.get("problem"), mode="check", created=None, chash="", ehash="", fmt="old",
                speedup=s.get("speedup"), passed=s.get("passed"), status=s.get("status"), cases=cases, src=name or "",
                recs=[{"case": x["case"], "median_ns": x["median_ns"], "baseline_ns": x["baseline_ns"]} for x in cases])


def refresh_archive():
    out = subprocess.run(["git", "ls-tree", "-r", "--full-tree", "--name-only", ARCHIVE_COMMIT], cwd=KB, capture_output=True, text=True, check=True).stdout
    names = [n for n in out.splitlines() if re.match(r"experiments/top50/profiles/decoder_.*\.json$", n)]
    rows = []
    for n in sorted(names):
        raw = subprocess.run(["git", "show", f"{ARCHIVE_COMMIT}:{n}"], cwd=KB, capture_output=True).stdout.decode("utf-8-sig", "replace")
        try:
            j = json.loads(raw)
        except Exception:
            continue
        if "benchmarks" not in j or (j.get("summary") or {}).get("speedup") is None:
            continue
        o = norm_old(j, os.path.basename(n))
        rows.append(dict(id=o["id"], src=o["src"], speedup=o["speedup"], passed=o["passed"], status=o["status"],
                         cases=[dict(key=c["key"], case=c["case"], median_ns=c["median_ns"], baseline_ns=c["baseline_ns"], sol_ns=c["sol_ns"],
                                     fam=families(c["report"]), prof_total=sum(families(c["report"]).values())) for c in o["cases"]]))
    with open(ARCHIVE_CACHE, "w", encoding="utf-8") as f:
        json.dump(dict(commit=ARCHIVE_COMMIT, jobs=rows), f, separators=(",", ":"))
    return len(rows)


def load_old():
    if not os.path.isfile(ARCHIVE_CACHE):
        refresh_archive()
    d = jload(ARCHIVE_CACHE)
    jobs = []
    for r in d["jobs"]:
        cases = [dict(key=c["key"], case=c["case"], median_ns=c["median_ns"], baseline_ns=c["baseline_ns"], sol_ns=c.get("sol_ns"),
                      report="", fam=c.get("fam", {})) for c in r["cases"]]
        jobs.append(dict(id=r["id"], board="muse_decode_layer", mode="check", created=None, chash="", ehash="", fmt="old",
                         speedup=r["speedup"], passed=r["passed"], status=r["status"], cases=cases, src=r["src"],
                         recs=[{"case": c["case"], "median_ns": c["median_ns"], "baseline_ns": c["baseline_ns"]} for c in cases]))
    return jobs


def load_all():
    """Return (accepted_jobs, rejected[(id, reason)]). Decode board only."""
    jobs, rej, seen = [], [], set()
    for d in RUN_DIRS:
        for p in sorted(glob.glob(os.path.join(d, "*.json"))):
            try:
                o = norm_new(p)
            except Exception as e:  # noqa: BLE001
                rej.append((os.path.basename(p), f"unreadable: {e}"))
                continue
            if o["board"] != "muse_decode_layer":
                continue
            if o["id"] in seen:
                continue
            seen.add(o["id"])
            jobs.append(o)
    jobs += load_old()
    ok = []
    for o in jobs:
        why = None
        keys = [c["key"] for c in o["cases"]]
        if not o["cases"]:
            why = "no benchmark records (duplicate/failed job)"
        elif sorted(keys) != sorted(SHAPES):
            why = f"incomplete shapes ({len(keys)}/12)"
        elif o["speedup"] is None or not o["passed"]:
            why = "no passing judge speedup"
        else:
            sc, _ = ms.score(o["recs"])
            if abs(sc - float(o["speedup"])) > TOL:
                why = f"replay mismatch {sc:.4f} vs {float(o['speedup']):.4f}"
            else:
                o["score"] = sc
        if why:
            rej.append((o["id"] or o["src"], why))
        else:
            ok.append(o)
    return ok, rej


LIN = None


def lineage():
    global LIN
    if LIN is None:
        LIN = jload(os.path.join(DATA, "lineage.json"))["jobs"]
    return LIN


def label(o):
    L = lineage().get(o["id"])
    if L:
        return L
    if o["fmt"] == "old":
        m = re.match(r"decoder_(p\d+\w*?)_", o["src"]) or re.match(r"decoder_(\w+?)\.", o["src"])
        p = m.group(1) if m else "p?"
        return dict(path="old_" + p, group="old/" + p, parent=None, kind="clean", era="old", note="old archive")
    return dict(path="unlabelled", group="unlabelled", parent=None, kind="clean", era="keell", note="")


def order_key(o):
    if o["fmt"] == "old":
        m = re.search(r"decoder_p(\d+)", o["src"])
        return (0, int(m.group(1)) if m else 0, o["id"])
    return (1, o["created"] or 0, o["id"])


# ----------------------------------------------------------------------------- backtest / rows
def cmd_backtest(write=True):
    jobs, rej = load_all()
    jobs.sort(key=order_key)
    n_new = sum(1 for o in jobs if o["fmt"] == "new")
    n_old = len(jobs) - n_new
    say("== BACKTEST: score replay (sum(w*baseline)/sum(w*median) per load, geometric mean over 6 loads) ==")
    say(f"accepted {len(jobs)} Decode jobs ({n_new} new-format records, {n_old} old-archive format); rejected {len(rej)}")
    failed = jload(os.path.join(DATA, "lineage.json")).get("failed_jobs", {})
    for r in rej:
        say(f"  rejected {r[0]}: {r[1]}" + (f"  [{failed[r[0]]}]" if r[0] in failed else ""))
    for k, v in failed.items():
        if k not in [r[0] for r in rej]:
            say(f"  not a Decode timing job {k}: {v}")
    mx = max(abs(o["score"] - float(o["speedup"])) for o in jobs)
    say(f"replay: max |calc - judge| = {mx:.2e} over {len(jobs)} jobs (tolerance {TOL})")
    # whole-repo replay across all boards through muse_score itself
    n = bad = 0
    paths = []
    for d in RUN_DIRS:
        paths += glob.glob(os.path.join(d, "*.json"))
    ids = set()
    for p in paths:
        try:
            j, res, recs = ms.load_job(p)
        except Exception:
            continue
        if not recs or res.get("speedup") is None or j.get("id") in ids:
            continue
        ids.add(j.get("id"))
        n += 1
        bad += abs(ms.score(recs)[0] - float(res["speedup"])) > TOL
    say(f"all-board replay via muse_score.score: {n} unique jobs, {bad} mismatches")
    say("")
    say(f"{'job':9}{'fmt':4}{'mode':6}{'score':>8}  {'kind':10}{'path':14}{'group':24}")
    for o in jobs:
        L = label(o)
        say(f"{o['id']:9}{o['fmt']:4}{o['mode']:6}{o['score']:8.4f}  {L['kind']:10}{L['path']:14}{L['group']:24}")
    if write:
        rows, frows = [], []
        for o in jobs:
            L = label(o)
            for c in o["cases"]:
                med, base = c["median_ns"] / 1e3, c["baseline_ns"] / 1e3
                rows.append(dict(board="muse_decode_layer", evaluator=o["ehash"], created=o["created"] or "", job=o["id"], mode=o["mode"], src_hash=o["chash"],
                                 parent=L.get("parent") or "", path=L["path"], group=L["group"], era=L["era"], kind=L["kind"], shape=c["key"],
                                 weight=c["case"].get("weight"), median_us=round(med, 3), baseline_us=round(base, 3), speedup=round(base / med, 4),
                                 sol_us=round(c["sol_ns"] / 1e3, 3) if c.get("sol_ns") else "", job_score=round(o["score"], 4)))
                fam = c.get("fam") or families(c.get("report", ""))
                for f, t in sorted(fam.items()):
                    frows.append(dict(job=o["id"], path=L["path"], group=L["group"], shape=c["key"], family=f, kernel_us=round(t, 2)))
        for name, rr in (("rows.csv", rows), ("family_rows.csv", frows)):
            with open(os.path.join(HERE, name), "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rr[0].keys()))
                w.writeheader()
                w.writerows(rr)
        say("")
        say(f"wrote rows.csv ({len(rows)} job x shape rows) and family_rows.csv ({len(frows)} job x shape x kernel-family rows)")
    return jobs


# ----------------------------------------------------------------------------- noise
def noise_stats(jobs):
    kb1 = [o for o in jobs if label(o)["path"] == "kb1"]
    stat = {}
    for s in SHAPES:
        v = [c["median_ns"] / 1e3 for o in kb1 for c in o["cases"] if c["key"] == s]
        sp = [c["baseline_ns"] / c["median_ns"] for o in kb1 for c in o["cases"] if c["key"] == s]
        stat[s] = dict(n=len(v), mean=st.mean(v), sd=st.stdev(v), cv=st.stdev(v) / st.mean(v), sp_cv=st.stdev(sp) / st.mean(sp))
    return kb1, stat


def cmd_noise(jobs):
    kb1, stat = noise_stats(jobs)
    sc = [o["score"] for o in kb1]
    say("== NOISE FLOOR: identical default path (kb v1; every gated wrapper whose gate adopted 0 arms ran this code) ==")
    say(f"{len(kb1)} jobs, all path=kb1: score mean {st.mean(sc):.4f} sd {st.stdev(sc):.4f} range [{min(sc):.4f}, {max(sc):.4f}]  (sd = {st.stdev(sc) / st.mean(sc) * 100:.2f}% of score)")
    say(f"{'shape':6}{'mean us':>9}{'sd us':>8}{'cv%':>6}{'cv% of speedup':>16}")
    for s in SHAPES:
        t = stat[s]
        say(f"{s:6}{t['mean']:9.1f}{t['sd']:8.2f}{t['cv'] * 100:6.2f}{t['sp_cv'] * 100:16.2f}")
    # family noise
    say("per-family kernel-time sd across the same jobs (profile reports, us):")
    for fam in ("moe", "dense", "attention", "glue_route"):
        cells = []
        for s in SHAPES:
            v = [families(c["report"]).get(fam, 0.0) for o in kb1 if o["fmt"] == "new" for c in o["cases"] if c["key"] == s]
            v = [x for x in v if x > 0]
            if len(v) >= 3:
                cells.append(f"{s}:{st.mean(v):.1f}+-{st.stdev(v):.1f}")
        say(f"  {fam:11}" + " ".join(cells))
    say("same-code pairs in old archive (same revision, two jobs): " + ", ".join(
        f"{a}-{b}: {abs(x - y):.4f}" for a, b, x, y in [("p7", "p7b", 2.0806, 2.0667), ("p10", "p10b", 2.097, 2.1043), ("p6", "p6b", 2.0783, 2.0682), ("p8", "p8b", 2.0671, 2.0776)]))
    return stat


# ----------------------------------------------------------------------------- unchanged-parent predictor
def cmd_persist(jobs):
    say("== UNCHANGED-PARENT PREDICTOR (candidate score := parent score) vs running group mean ==")
    for era, name in (("keell", "kb1 path, time order"), ("old", "old archive, revision order")):
        seq = [o for o in jobs if (label(o)["path"] == "kb1" if era == "keell" else o["fmt"] == "old")]
        sc = [o["score"] for o in seq]
        e_par = [abs(sc[i] - sc[i - 1]) for i in range(1, len(sc))]
        e_mean = [abs(sc[i] - st.mean(sc[:i])) for i in range(2, len(sc))]
        say(f"  {name}: n={len(sc)}  MAE vs previous job {st.mean(e_par):.4f}x   MAE vs running mean {st.mean(e_mean):.4f}x   "
            f"(sd of scores {st.stdev(sc):.4f})")
    # per-shape persistence in log-latency
    kb1 = [o for o in jobs if label(o)["path"] == "kb1"]
    errs = {s: [] for s in SHAPES}
    for i in range(1, len(kb1)):
        a = {c["key"]: c for c in kb1[i - 1]["cases"]}
        b = {c["key"]: c for c in kb1[i]["cases"]}
        for s in SHAPES:
            errs[s].append(abs(math.log(b[s]["median_ns"] / a[s]["median_ns"])))
    say("  per-shape |log(child/parent)| for identical code (median): " + " ".join(f"{s}:{st.median(errs[s]) * 100:.1f}%" for s in SHAPES))
    say("  => any effect smaller than ~1-2% per shape (~1-3 us at d1, 3-6 us at d32/v16) is invisible in ONE judged job.")


# ----------------------------------------------------------------------------- sensitivity
_BASE = {}


def anchor():
    if not _BASE:
        p = ms.find_job(ANCHOR, RUN_DIRS)
        j, res, recs = ms.load_job(p)
        _BASE["recs"] = recs
        _BASE["score"], _BASE["ratios"] = ms.score(recs)
        _BASE["med"] = {ms.key_of(r["case"]): r["median_ns"] / 1e3 for r in recs}
        _BASE["sol"] = {ms.key_of(r["case"]): r["speed_of_light_ns"] / 1e3 for r in recs if r.get("speed_of_light_ns")}
        _BASE["w"] = {ms.key_of(r["case"]): r["case"].get("weight") for r in recs}
    return _BASE


def score_with(save):
    a = anchor()
    return ms.score(a["recs"], save_us=save)[0]


def sens(shape):
    """d score / d us for a 1 us saving on this shape alone (anchor job)."""
    a = anchor()
    return score_with({shape: 1.0}) - a["score"]


# ----------------------------------------------------------------------------- transfer
def fit_beta(xs, ys, ws, lam=0.0):
    num = sum(w * x * y for x, y, w in zip(xs, ys, ws))
    den = sum(w * x * x for x, w in zip(xs, ws)) + lam
    return num / den if den > 0 else 0.0


def group_weights(rows):
    cnt = {}
    for r in rows:
        cnt[r["group"]] = cnt.get(r["group"], 0) + 1
    return [1.0 / cnt[r["group"]] for r in rows]


def wmean(vals, ws):
    return sum(v * w for v, w in zip(vals, ws)) / sum(ws)


def derive_arms():
    d = jload(os.path.join(DATA, "transfer_arms.json"))
    ref = d["layer_ref_us"]
    arms = []
    for a in d["arms"]:
        r = dict(a)
        rf = ref[a["era"]][a["shape"]]
        r["ref"] = rf
        if "layer_us" not in a and "layer_ratio" in a:
            r["layer_us"] = (1.0 - a["layer_ratio"]) * rf
        arms.append(r)
    return arms, d["board_arms"]


def eval_pair(rows, xk, yk, title, cap=20.0, thr_frac=0.01, board=False):
    rows = [r for r in rows if r.get(xk) is not None and r.get(yk) is not None]
    if not rows:
        return None
    ws = group_weights(rows)
    xs = [r[xk] for r in rows]
    ys = [r[yk] for r in rows]
    groups = sorted(set(r["group"] for r in rows))
    # leave-one-group-out shrink slope (through origin), clipped to [0, 1]
    pred_s = []
    for r in rows:
        tr = [(q[xk], q[yk], w) for q, w in zip(rows, ws) if q["group"] != r["group"]]
        b = fit_beta([t[0] for t in tr], [t[1] for t in tr], [t[2] for t in tr]) if tr else 0.0
        pred_s.append(max(0.0, min(1.0, b)) * r[xk])
    b_all = fit_beta(xs, ys, ws)
    res = {}
    for nm, pr in (("zero-transfer", [0.0] * len(rows)), ("full-transfer", xs), ("LOGO-shrunk", pred_s)):
        err = [abs(p - y) for p, y in zip(pr, ys)]
        res[nm] = dict(mae=wmean(err, ws), mae_cap=wmean([min(e, cap) for e in err], ws), med=st.median(err),
                       score_mae=wmean([e * abs(sens(r["shape"])) if not board else float("nan") for e, r in zip(err, rows)], ws) if not board else float("nan"))
    # sign accuracy beyond noise
    thr = [(thr_frac * r["ref"]) if "ref" in r else 1.0 for r in rows]
    beyond = [i for i, y in enumerate(ys) if abs(y) >= thr[i]]
    hit_full = sum(1 for i in beyond if (xs[i] > 0) == (ys[i] > 0))
    hit_pos = sum(1 for i in beyond if ys[i] > 0)
    say(f"-- {title}: n={len(rows)} rows, {len(groups)} revision groups; group-weighted")
    say(f"   {'predictor':14}{'MAE us':>9}{'MAE(cap20)':>12}{'median AE':>11}{'score MAE':>11}")
    for nm, v in res.items():
        sm = "   n/a" if math.isnan(v["score_mae"]) else f"{v['score_mae']:11.5f}"
        say(f"   {nm:14}{v['mae']:9.2f}{v['mae_cap']:12.2f}{v['med']:11.2f}{sm:>11}")
    if not board:
        say(f"   rows beyond noise (|y|>=1% of layer): {len(beyond)}; sign of the claim (x>0 means faster) is right in {hit_full}/{len(beyond)}; "
            f"measured faster in {hit_pos}/{len(beyond)}")
    else:
        big = [i for i, y in enumerate(ys) if abs(y) >= 1.0]
        say(f"   rows with |y|>=1 us: {len(big)}; claim sign right in {sum(1 for i in big if (xs[i] > 0) == (ys[i] > 0))}/{len(big)}")
    rng = random.Random(7)
    bs = []
    for _ in range(2000):
        gs = [rng.choice(groups) for _ in groups]
        sub, sw = [], []
        for g in gs:
            m = [(r, w) for r, w in zip(rows, ws) if r["group"] == g]
            for r, w in m:
                sub.append(r)
                sw.append(w)
        bs.append(fit_beta([r[xk] for r in sub], [r[yk] for r in sub], sw))
    bs.sort()
    lo, hi = bs[int(0.1 * len(bs))], bs[int(0.9 * len(bs))]
    say(f"   fitted transfer slope y = b*x: b={b_all:.2f}  (group-bootstrap 80% interval [{lo:.2f}, {hi:.2f}]); "
        f"{'interval contains 0 -> not distinguishable from zero transfer' if lo <= 0 <= hi else 'interval excludes 0'}")
    pos = [r for r in rows if r[xk] > 0]
    hit = sum(1 for r in pos if r[yk] >= 0.5 * r[xk] and r[yk] >= 0.5)
    say(f"   claims>0: {len(pos)} rows, {hit} realised >=50% of the claim and >=0.5 us (empirical hit rate {hit}/{len(pos)}; not a calibrated probability)")
    return dict(beta=b_all, lo=lo, hi=hi, hit=hit, n=len(pos), rows=rows, ws=ws, res=res)


def topk(rows, xk, yk, k=3):
    rows = [r for r in rows if r.get(xk) is not None and r.get(yk) is not None]
    if len(rows) < 4:
        return None
    by = sorted(rows, key=lambda r: -r[xk])[:k]
    allm = st.mean(r[yk] for r in rows)
    orc = st.mean(sorted((r[yk] for r in rows), reverse=True)[:k])
    return st.mean(r[yk] for r in by), allm, orc


def cmd_transfer(jobs):
    arms, board = derive_arms()
    say("== TRANSFER: does a claimed/isolated gain survive into the full layer? (positive = faster, us) ==")
    say("claims = design/ledger numbers made BEFORE timing (pre); isolated = standalone paired A/B (iso); layer = in-layer ABBA or judge-check delta.")
    say("NEGATIVE CONTROLS inside the data: 7bae45c5 (pre +5-7 us, layer -1.2..-11.4 us) and 74e3e792 (pre -7.5 us at bs16, measured +99.5 us slower).")
    res = {}
    res["P2L"] = eval_pair(arms, "pre_us", "layer_us", "pre-measurement claim -> full-layer delta")
    say("")
    res["I2L"] = eval_pair(arms, "iso_us", "layer_us", "isolated A/B gain -> full-layer delta")
    say("")
    both = [dict(r, group=r["group"]) for r in arms if r.get("pre_us") is not None and r.get("iso_us") is not None]
    allp2i = both + [dict(b, ref=1e9) for b in board]
    res["P2I"] = eval_pair(allp2i, "pre_us", "iso_us", "pre-measurement claim -> isolated/board A/B (includes VMS iterations; 74e3e792 is the 2x-slower row)", board=True)
    say("")
    say("-- old-to-keell rebase shift (fit slope on one era, test on the other; group-weighted MAE us)")
    for xk, yk in (("pre_us", "layer_us"), ("iso_us", "layer_us")):
        for tr_e, te_e in (("old", "keell"), ("keell", "old")):
            tr = [r for r in arms if r["era"] == tr_e and r.get(xk) is not None and r.get(yk) is not None]
            te = [r for r in arms if r["era"] == te_e and r.get(xk) is not None and r.get(yk) is not None]
            if len(tr) < 3 or len(te) < 3:
                say(f"   {xk}->{yk} train {tr_e} ({len(tr)}) test {te_e} ({len(te)}): too few rows")
                continue
            b = fit_beta([r[xk] for r in tr], [r[yk] for r in tr], group_weights(tr))
            b = max(0.0, min(1.0, b))
            w = group_weights(te)
            e_s = wmean([min(abs(b * r[xk] - r[yk]), 20) for r in te], w)
            e_0 = wmean([min(abs(r[yk]), 20) for r in te], w)
            e_f = wmean([min(abs(r[xk] - r[yk]), 20) for r in te], w)
            say(f"   {xk}->{yk} train {tr_e} (n={len(tr)}, b={b:.2f}) test {te_e} (n={len(te)}): MAE(cap20) shrunk {e_s:.2f} vs zero {e_0:.2f} vs full {e_f:.2f}")
    say("")
    say("-- top-3 selection quality (rank arms by the claim; mean realised layer gain of the top 3 vs all arms vs oracle top 3)")
    for xk, nm in (("pre_us", "by pre claim"), ("iso_us", "by isolated gain")):
        for era in ("old", "keell"):
            t = topk([r for r in arms if r["era"] == era], xk, "layer_us")
            say(f"   {nm}, {era}: " + ("too few arms" if not t else f"top3 {t[0]:+.2f} us  all-arms mean {t[1]:+.2f} us  oracle top3 {t[2]:+.2f} us"))
    say("")
    say("Reading: slope intervals and LOGO errors above decide whether the data support ANY source-only forecast; if the zero-transfer row")
    say("is as good as the shrunk row, the honest forecast for an untimed claim is 'zero, range 0..claim', not a point estimate.")
    return res


# ----------------------------------------------------------------------------- gate -> judge
def job_index(jobs):
    return {o["id"]: o for o in jobs}


def cmd_gate(jobs):
    d = jload(os.path.join(DATA, "gate_arms.json"))["arms"]
    idx = job_index(jobs)
    rows = []
    for a in d:
        ch = idx.get(a["child"])
        ps = [idx[p] for p in a["parents"] if p in idx]
        if not ch or not ps:
            continue
        cm = {c["key"]: c for c in ch["cases"]}[a["shape"]]
        pm = [{c["key"]: c for c in p["cases"]}[a["shape"]] for p in ps]
        parent_med = st.mean(c["median_ns"] for c in pm) / 1e3
        parent_base = st.mean(c["baseline_ns"] for c in pm) / 1e3
        child_med = cm["median_ns"] / 1e3
        child_norm = child_med * parent_base / (cm["baseline_ns"] / 1e3)  # child time on the parent's baseline scale
        rows.append(dict(id=a["id"], group=a["child"], shape=a["shape"], ref=parent_med, gate_us=a["gate_us"],
                         raw_us=parent_med - child_med, norm_us=parent_med - child_norm))
    say("== GATE -> JUDGE: adopted in-prepare full-layer gate gains vs the judge's parent->child delta (old archive, us, positive = faster) ==")
    say(f"{'arm':10}{'shape':6}{'gate':>7}{'judge raw':>11}{'judge norm':>12}")
    for r in rows:
        say(f"{r['id']:10}{r['shape']:6}{r['gate_us']:7.2f}{r['raw_us']:11.2f}{r['norm_us']:12.2f}")
    ev = {}
    for k in ("raw_us", "norm_us"):
        ev[k] = eval_pair(rows, "gate_us", k, f"gate gain -> judge delta ({k})")
        say("")
    # same-code reference: sd of the judge delta when nothing changed
    old = [o for o in jobs if o["fmt"] == "old"]
    sd = {}
    for s in SHAPES:
        v = [c["median_ns"] / 1e3 for o in old for c in o["cases"] if c["key"] == s]
        sd[s] = st.stdev(v)
    say("   old-archive per-shape sd of the judged median (us): " + " ".join(f"{s}:{sd[s]:.1f}" for s in SHAPES))
    say("   a parent->child delta therefore carries sd ~ sqrt(2)*that, e.g. d1S ~%.1f us vs gate claims of 1.4-6.7 us." % (math.sqrt(2) * sd["d1S"]))
    return ev


# ----------------------------------------------------------------------------- overlap / critical path
def cmd_overlap(jobs):
    kb1 = [o for o in jobs if label(o)["path"] == "kb1" and o["fmt"] == "new"]
    say("== CRITICAL-PATH ESTIMATE: per-kernel profile sum vs judged layer time (f2ab6d4c profile, all kb1 jobs averaged) ==")
    say("A profile kernel's time includes waits for the PDL-overlapped predecessor, so sum(kernels) over-counts; judged/sum = fraction of a kernel saving that can")
    say("reach the critical path at most. Family columns are mean kernel us across kb1-path jobs.")
    say(f"{'shape':6}{'judged':>8}{'profSum':>9}{'ratio':>7}{'moe':>7}{'dense':>7}{'attn':>7}{'glue':>7}{'sol':>8}{'judged/sol':>11}")
    out = {}
    for s in SHAPES:
        cs = [c for o in kb1 for c in o["cases"] if c["key"] == s]
        med = st.mean(c["median_ns"] for c in cs) / 1e3
        fam = {}
        for f in ("moe", "dense", "attention", "glue_route"):
            fam[f] = st.mean(families(c["report"]).get(f, 0.0) for c in cs)
        tot = sum(fam.values()) + st.mean(families(c["report"]).get("other", 0.0) for c in cs)
        sol = st.mean(c["sol_ns"] for c in cs) / 1e3
        out[s] = dict(med=med, tot=tot, fam=fam, sol=sol)
        say(f"{s:6}{med:8.1f}{tot:9.1f}{med / tot:7.2f}{fam['moe']:7.1f}{fam['dense']:7.1f}{fam['attention']:7.1f}{fam['glue_route']:7.1f}{sol:8.1f}{med / sol:11.2f}")
    return out


# ----------------------------------------------------------------------------- need (2.37)
JOEY = {"d1S": 2.70, "d1F": 2.56, "d16S": 2.33, "d16F": 2.28, "d32S": 1.86, "d32F": 2.06, "v1S": 2.78, "v1F": 2.48, "v16S": 2.26, "v16F": 2.21, "v32S": 2.40, "v32F": 2.31}


def cmd_need():
    a = anchor()
    say(f"== WHAT 2.37 NEEDS (anchor {ANCHOR}: scored {a['score']:.4f}; muse_score what-ifs) ==")
    say("per-load speedups at anchor: " + " ".join(f"{k}:{v:.4f}" for k, v in a["ratios"].items()))
    tgt = 2.37

    def uni(x):
        return score_with({s: x for s in SHAPES})

    lo, hi = 0.0, 40.0
    for _ in range(50):
        m = (lo + hi) / 2
        lo, hi = (m, hi) if uni(m) < tgt else (lo, m)
    say(f"uniform saving on all 12 shapes: {lo:.1f} us/layer reaches {tgt} (x{tgt / a['score'] - 1:+.1%} needed); -3/-5/-8 us -> {uni(3):.4f}/{uni(5):.4f}/{uni(8):.4f}")
    sc = {}
    for s in SHAPES:
        r = [x for x in a["recs"] if ms.key_of(x["case"]) == s][0]
        sc[s] = (r["baseline_ns"] / 1e3) / (r["median_ns"] / 1e3) / JOEY[s]  # ours speedup / Joey speedup  (=time scale to match Joey ratio)
    match = {s: min(1.0, sc[s]) for s in SHAPES}  # only where Joey leads
    say(f"match Joey's ratio on every shape where he leads (his table, our baseline): {ms.score(a['recs'], scale=match)[0]:.4f}  (shapes: "
        + " ".join(f"{s}:{(1 - match[s]) * 100:.1f}%" for s in SHAPES if match[s] < 1) + ")")
    brief = {"d16S": 10.0 + 3.5, "d16F": 8.0 + 3.5, "d1S": 4.4, "d1F": 4.4, "v1S": 2.4, "v1F": 2.4}
    say(f"  task-brief scenario (d16 -10/-8 + dense 4.4/2.4/3.5 on both layer types, full transfer): {score_with(brief):.4f}")
    cands = jload(os.path.join(DATA, "candidates.json"))["candidates"]
    meas, des = {}, {}
    for k in cands:
        if k["id"] in ("DG-T1", "DG-T8", "DG-T16", "VMS-16", "VML-32"):
            for s, v in k["gross_us"].items():
                meas[s] = meas.get(s, 0) + v
            for s, v in k["design_us"].items():
                des[s] = des.get(s, 0) + v
    def extra_uniform(base):
        lo, hi = 0.0, 40.0
        for _ in range(50):
            m = (lo + hi) / 2
            d = {sh: base.get(sh, 0) + m for sh in SHAPES}
            lo, hi = (m, hi) if score_with(d) < tgt else (lo, m)
        return lo
    say(f"  extra uniform saving still needed on all 12 shapes: after design-number lanes {extra_uniform(des):.1f} us, after measured-so-far lanes {extra_uniform(meas):.1f} us, from scratch {extra_uniform({}):.1f} us")
    say(f"  the five current lane rows at DESIGN numbers: {score_with(des):.4f};  at numbers MEASURED so far (8ffd6871, c5800fa6): {score_with(meas):.4f} "
        f"(+{score_with(meas) - a['score']:.4f}; gap to {tgt} is x{tgt / score_with(meas) - 1:+.2%})")
    lanes = {
        "VMS d16 (-10.4 S / -7.6 F)": {"d16S": 10.4, "d16F": 7.6},
        "VMS d16 + VML d32 (-3.3 S/-1.8 F)": {"d16S": 10.4, "d16F": 7.6, "d32S": 3.3, "d32F": 1.8},
        "dense T=1/8/16 (4.4/2.4/3.5, both layer types)": {"d1S": 4.4, "d1F": 4.4, "v1S": 2.4, "v1F": 2.4, "d16S": 3.5, "d16F": 3.5},
    }
    both = {}
    for k, v in lanes.items():
        say(f"  {k}: {score_with(v):.4f}")
    for v in (lanes["VMS d16 + VML d32 (-3.3 S/-1.8 F)"], lanes["dense T=1/8/16 (4.4/2.4/3.5, both layer types)"]):
        for s, x in v.items():
            both[s] = both.get(s, 0) + x
    full = score_with(both)
    say(f"  all lanes added, full transfer: {full:.4f}  -> remaining gap to {tgt}: x{tgt / full - 1:+.2%}")
    rem = {}
    lo, hi = 0.0, 40.0
    for _ in range(50):
        m = (lo + hi) / 2
        d = dict(both)
        for s in SHAPES:
            d[s] = d.get(s, 0) + m
        lo, hi = (m, hi) if score_with(d) < tgt else (lo, m)
    say(f"  extra uniform saving on all 12 shapes after the lanes: {lo:.1f} us/layer (weights: sliding x46, full x16 per load)")
    att = 0.031 * 16.1  # attn board v32S 1h-s4 ratio 0.968/0.969 on the 16.1 us attn_kernel (profile v32S)
    say(f"  coverage of the verify-32 sliding one-head attention win in Decode: -{att:.2f} us on v32S only (q1 sliding shapes d16S/d32S showed no 1h win) -> {score_with({'v32S': att}):.4f}"
        f" (+{score_with({'v32S': att}) - a['score']:.5f}); even 1 us on every sliding shape is only {score_with({s: 1.0 for s in SHAPES if s.endswith('S')}):.4f}")
    say("  sensitivity (score per 1 us on one shape): " + " ".join(f"{s}:{sens(s):.5f}" for s in SHAPES))
    return full


# ----------------------------------------------------------------------------- chase
def cmd_chase(tres):
    c = jload(os.path.join(DATA, "candidates.json"))
    a = anchor()
    beta = {k: max(0.0, min(1.0, tres[k]["beta"])) if tres.get(k) else 0.0 for k in tres}
    beta["gate"] = max(0.0, min(1.0, tres["GATE"]["beta"])) if tres.get("GATE") else 0.0
    cls_map = {"pre": "P2L", "iso": "I2L", "gate": "GATE"}
    say("== CHASE: ranked by expected score gain per check ==")
    say(f"anchor {ANCHOR} = {a['score']:.4f}. gross = design/ledger claim; cap = documented headroom (floor); net_full = min(gross,cap) (full transfer, ceiling);")
    say("net_hat = shrunk slope of the claim's evidence class (see TRANSFER/GATE; clipped to [0,1]); hit = empirical rate (k+1)/(n+2) that a claim of that class realised >=50% of itself.")
    say("E[gain/check] = hit * (score_full - anchor) / checks. 'Conditional' score = score if the claim materialises at full transfer.")
    say("")
    rows = []
    for k in c["candidates"]:
        gross, cap = k["gross_us"], k.get("cap_us", {})
        g = {s: min(v * k.get("haircut", 1.0), cap.get(s, 1e9)) for s, v in gross.items()}
        cl = k["evidence"]
        t = tres.get(cls_map[cl]) if cl in cls_map else None
        b = beta.get(cls_map[cl] if cl != "gate" else "gate", 0.0)
        hit = ((t["hit"] + 1) / (t["n"] + 2)) if t else 0.1
        full = score_with(g)
        hat = score_with({s: v * b for s, v in g.items()})
        egain = hit * (full - a["score"]) / k["checks"]
        ceil = score_with({s: cap.get(s, v) for s, v in gross.items()})
        if k.get("killed"):
            egain = 0.0
        lim = c.get("_limits", {}).get("max_dyn_smem_B", 232448)
        res = k.get("resources")
        feas = "n/a (not built)" if not res else ("PASS" if res["dyn_smem_B"] <= lim and res["spill_B"] == 0 else "FAIL")
        rows.append(dict(k=k, g=g, ceil=ceil, full=full, hat=hat, hit=hit, b=b, egain=egain, ealt=(0 if k.get("killed") else 0.25 * (full - a["score"]) / k["checks"]),
                         per_check=(full - a["score"]) / k["checks"], feas=feas))
    rows.sort(key=lambda r: -r["egain"])
    for i, r in enumerate(rows, 1):
        k = r["k"]
        gs = " ".join(f"{s}:{k['gross_us'][s]:.1f}->{r['g'][s]:.1f}" for s in k["gross_us"])
        ds = " ".join(f"{s}:{v:.1f}" for s, v in k["design_us"].items())
        say(f"#{i} {k['id']}: {k['name']}  [{k['status']}]" + ("  ** KILLED by existing data **" if k.get("killed") else ""))
        say(f"    design claim / documented headroom us: {ds}   resources feasibility (smem<=227KiB, 0 spills): {r['feas']}")
        say(f"    parent: {k['parent']}   change: {k['change']}")
        say(f"    loads: {k['loads']}   mechanism: {k['mechanism']}")
        say(f"    gross->net_full us per shape: {gs}")
        say(f"    net_hat us per shape (class slope b={r['b']:.2f}): " + " ".join(f"{s}:{v * r['b']:.2f}" for s, v in r["g"].items()) + f"   ceiling (net=cap) score {r['ceil']:.4f}")
        say(f"    conditional score {r['full']:.4f} (+{r['full'] - a['score']:.4f}); at class slope b={r['b']:.2f}: {r['hat']:.4f} (+{r['hat'] - a['score']:.4f}); "
            f"checks {k['checks']}; hit {r['hit']:.2f}; E[gain/check] {r['egain']:.5f} (optimistic flat hit=0.25: {r['ealt']:.5f})")
        say(f"    evidence [{k['evidence']}]: {k['evidence_text']}")
        say(f"    kill: {k['kill']}")
    return rows


def cmd_all(out):
    jobs = cmd_backtest()
    say("")
    cmd_noise(jobs)
    say("")
    cmd_persist(jobs)
    say("")
    cmd_overlap(jobs)
    say("")
    tres = cmd_transfer(jobs)
    say("")
    gres = cmd_gate(jobs)
    tres["GATE"] = gres["norm_us"]
    say("")
    cmd_need()
    say("")
    cmd_chase(tres)
    if out:
        with open(out, "w", encoding="utf-8") as f:
            f.write("\n".join(OUT) + "\n")
        print(f"[wrote {out}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["backtest", "noise", "transfer", "gate", "overlap", "need", "chase", "all", "refresh-archive"])
    ap.add_argument("--out")
    ap.add_argument("--no-write", action="store_true")
    a = ap.parse_args()
    if a.cmd == "refresh-archive":
        print("archive jobs cached:", refresh_archive())
        return
    if a.cmd == "all":
        cmd_all(a.out or os.path.join(HERE, "BACKTEST.txt"))
        return
    jobs = cmd_backtest(write=not a.no_write) if a.cmd == "backtest" else load_all()[0]
    jobs.sort(key=order_key)
    if a.cmd == "noise":
        cmd_noise(jobs)
    elif a.cmd == "transfer":
        cmd_transfer(jobs)
    elif a.cmd == "gate":
        cmd_gate(jobs)
    elif a.cmd == "overlap":
        cmd_overlap(jobs)
    elif a.cmd == "need":
        cmd_need()
    elif a.cmd == "chase":
        t = cmd_transfer(jobs)
        t["GATE"] = cmd_gate(jobs)["norm_us"]
        cmd_chase(t)


if __name__ == "__main__":
    main()
