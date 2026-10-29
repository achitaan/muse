"""Offline analysis of archived leaderboard snapshots (no network, no GPU).

Tests candidate aggregation formulas for the overall board against the
archived overall ranks, then estimates how much each track would have to
improve for the account to reach a given overall rank.
"""
import json, math, sys, itertools
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SNAP = ROOT / "results" / "model-handoff-20261029"
TRACKS = ["muse_bf16_proj", "muse_attention_decode", "muse_moe_w4a16",
          "muse_decode_layer", "muse_prefill_layer"]


def load():
    boards = {}
    for t in TRACKS:
        r = json.loads((SNAP / f"{t}-board.json").read_text())["response"]
        boards[t] = {row["team"]: row for row in r["rows"]}
    ov = json.loads((SNAP / "overall.json").read_text())["response"]
    return boards, ov


def main():
    boards, ov = load()
    overall = {r["team"]: r for r in ov["rows"]}
    # baseline SGLang time implied by value*speedup
    for t in TRACKS:
        b = boards[t]
        base = [r["value"] * r["speedup"] for r in b.values() if r.get("speedup")]
        base.sort()
        print(t, "n", len(b), "implied baseline value median", base[len(base)//2],
              "min", base[0], "max", base[-1])
    return boards, overall


BASE = {"muse_bf16_proj": 4830.8, "muse_attention_decode": 3766.3,
        "muse_moe_w4a16": 13448.5, "muse_decode_layer": 22620.3,
        "muse_prefill_layer": 387404.0}
OV = TRACKS[:4]


def metrics(boards, team, override=None):
    override = override or {}
    out = {}
    vals, ranks, logs = {}, {}, {}
    for t in OV:
        row = boards[t].get(team)
        sp = override.get(t, row["speedup"] if row else None)
        if sp is None:
            vals[t] = BASE[t]; logs[t] = 0.0
            ranks[t] = len(boards[t]) + 1
        else:
            vals[t] = BASE[t] / sp; logs[t] = math.log(sp)
            if t in override:
                ranks[t] = 1 + sum(1 for r in boards[t].values()
                                   if r["team"] != team and r["speedup"] > sp)
            else:
                ranks[t] = row["rank"]
    out["sum_time"] = sum(vals.values())
    out["sum_time_nodecode"] = sum(v for k, v in vals.items() if k != "muse_decode_layer")
    out["mean_rank"] = sum(ranks.values())
    out["neg_geo"] = -sum(logs.values())
    out["sum_inv"] = sum(1 / math.exp(l) for l in logs.values())
    return out


def kendall_like(order_true, score):
    # fraction of adjacent pairs (in true order) that score orders correctly
    ok = tot = 0
    for a, b in zip(order_true, order_true[1:]):
        tot += 1
        ok += score[a] <= score[b]
    return ok / tot


def evaluate(boards, overall):
    teams = sorted(overall, key=lambda t: overall[t]["rank"])
    m = {t: metrics(boards, t) for t in teams}
    for key in next(iter(m.values())):
        sc = {t: m[t][key] for t in teams}
        srt = sorted(teams, key=lambda t: sc[t])
        exact = sum(1 for i, t in enumerate(srt) if abs((i + 1) - overall[t]["rank"]) <= 0)
        print(f"{key:20s} adjacent-order agreement {kendall_like(teams, sc):.4f}"
              f" exact-rank hits {exact}/{len(teams)}")
    return m


def show(boards, ov):
    from collections import Counter
    rows = ov["rows"]
    print(Counter(r["solved"] for r in rows))
    def line(r):
        t = r["team"]; s = []
        for k in OV:
            x = boards[k].get(t)
            s.append(f"{x['rank']:5d}/{x['speedup']:.3f}" if x else "    -/-    ")
        print(r["rank"], r["solved"], t[:16].ljust(16), " ".join(s))
    for sl in (slice(0, 25), slice(40, 60), slice(195, 215)):
        for r in rows[sl]:
            line(r)
        print("...")


def decode_identity(boards, ov):
    dec = boards["muse_decode_layer"]
    same = diff = nodec = 0
    ex = []
    for r in ov["rows"]:
        d = dec.get(r["team"])
        if d is None:
            nodec += 1
            if len(ex) < 8:
                ex.append((r["rank"], r["solved"], r["team"],
                           {k: (boards[k][r["team"]]["speedup"] if r["team"] in boards[k] else None)
                            for k in TRACKS}))
        elif d["rank"] == r["rank"]:
            same += 1
        else:
            diff += 1
    print("overall rank == decode rank:", same, "differ:", diff, "no decode entry:", nodec)
    for e in ex:
        print("  no-decode example", e)


def brackets(boards, ov):
    """For teams w/o decode entry, bracket their overall metric by neighbours
    that have decode entries (assuming those teams' metric = decode speedup)."""
    dec = boards["muse_decode_layer"]
    rows = ov["rows"]
    # consistency: decode-entry teams appear in decode-speedup order?
    seq = [dec[r["team"]]["speedup"] for r in rows if r["team"] in dec]
    inv = sum(1 for a, b in zip(seq, seq[1:]) if b > a + 1e-9)
    print("decode-entry teams in overall order:", len(seq), "adjacent inversions:", inv)
    out = []
    for i, r in enumerate(rows):
        if r["team"] in dec:
            continue
        hi = next((dec[x["team"]]["speedup"] for x in reversed(rows[:i]) if x["team"] in dec), None)
        lo = next((dec[x["team"]]["speedup"] for x in rows[i + 1:] if x["team"] in dec), None)
        sp = {k: boards[k][r["team"]]["speedup"] if r["team"] in boards[k] else None for k in OV[:3]}
        out.append((r["rank"], r["solved"], hi, lo, sp))
    return out


def fit(out):
    import itertools
    best = None
    # composite: S = T / (sum_k w_k / s_k + w_other), w normalised; missing -> s=1
    names = OV[:3]
    grid = [x / 100 for x in range(0, 101, 2)]
    for wb in grid:
        for wa in grid:
            for wm in grid:
                wo = 1 - wb - wa - wm
                if wo < -1e-9:
                    continue
                bad = 0
                for rank, solved, hi, lo, sp in out:
                    t = wo + sum(w / (sp[n] or 1.0) for w, n in zip((wb, wa, wm), names))
                    s = 1 / t
                    if (hi is not None and s > hi + 5e-4) or (lo is not None and s < lo - 5e-4):
                        bad += 1
                if best is None or bad < best[0]:
                    best = (bad, wb, wa, wm, wo)
    print("best fit violations/of", best, len(out))
    return best


def rank_fit(boards, ov, use_max=True, step=0.02):
    import numpy as np
    rows = ov["rows"]
    teams = [r["team"] for r in rows]
    true = np.array([r["rank"] for r in rows], float)
    def arr(k):
        return np.array([boards[k][t]["speedup"] if t in boards[k] else 1.0 for t in teams])
    sb, sa, sm, sd = (arr(k) for k in OV)
    hasd = np.array([t in boards["muse_decode_layer"] for t in teams])
    ib, ia, im = 1 / sb, 1 / sa, 1 / sm
    best = None
    g = np.arange(0, 1.0001, step)
    for wb in g:
        for wa in g:
            for wm in g:
                wo = 1 - wb - wa - wm
                if wo < -1e-9:
                    continue
                comp = 1 / (wb * ib + wa * ia + wm * im + wo)
                met = np.where(hasd, np.maximum(sd, comp) if use_max else sd, comp)
                order = np.argsort(-met, kind="stable")
                pr = np.empty(len(met)); pr[order] = np.arange(1, len(met) + 1)
                err = np.abs(pr - true).mean()
                if best is None or err < best[0]:
                    best = (err, wb, wa, wm, wo)
    print("use_max", use_max, "best mean |rank err|", best)
    return best


def predict(boards, ov, w, team="achitaan", decode_override=None):
    import numpy as np
    rows = ov["rows"]
    teams = [r["team"] for r in rows]
    true = np.array([r["rank"] for r in rows], float)
    def arr(k):
        return np.array([boards[k][t]["speedup"] if t in boards[k] else 1.0 for t in teams])
    sb, sa, sm, sd = (arr(k) for k in OV)
    hasd = np.array([t in boards["muse_decode_layer"] for t in teams])
    wb, wa, wm, wo = w
    comp = 1 / (wb / sb + wa / sa + wm / sm + wo)
    met = np.where(hasd, np.maximum(sd, comp), comp)
    order = np.argsort(-met, kind="stable")
    pr = np.empty(len(met)); pr[order] = np.arange(1, len(met) + 1)
    err = np.abs(pr - true)
    print(f"weights {w}: mean |err| {err.mean():.2f}, median {np.median(err):.1f},"
          f" top-300 mean {err[:300].mean():.2f}, max {err.max():.0f}")
    i = teams.index(team)
    print(f"  {team}: decode {sd[i]:.4f} composite {comp[i]:.4f} predicted {pr[i]:.0f} actual {true[i]:.0f}")
    print(f"  teams whose composite exceeds decode: {(hasd & (comp > sd)).sum()} of {hasd.sum()}")
    for d in decode_override or []:
        rank = 1 + int((np.delete(met, i) > d).sum())
        print(f"  if {team} decode = {d:.4f}: predicted overall rank {rank}")


if __name__ == "__main__":
    boards, overall = main()
    evaluate(boards, overall)
    ov = load()[1]
    decode_identity(boards, ov)
    out = brackets(boards, ov)
    for o in out[:6]:
        print(o)
    if "--fit" in sys.argv:
        best = rank_fit(boards, ov, True, 0.01)
        rank_fit(boards, ov, False)
        predict(boards, ov, best[1:], decode_override=[2.05, 2.09, 2.12, 2.15, 2.18, 2.19, 2.25])
    dec = sorted((r["speedup"] for r in boards["muse_decode_layer"].values()), reverse=True)
    ovr = {r["team"]: r["rank"] for r in ov["rows"]}
    for target in (25, 50, 75, 100, 150, 206):
        # decode-entry team currently sitting at that overall rank
        t = next(r["team"] for r in ov["rows"] if r["rank"] == target)
        d = boards["muse_decode_layer"].get(t)
        print(f"overall rank {target}: team decode speedup {d['speedup'] if d else None}"
              f" -> needed gain over 1.9940 = {((d['speedup'] if d else 0)/1.9940-1)*100:.2f}%")
    # composite ceiling with fitted weights for our account
    wb, wa, wm, wo = 0.25, 0.11, 0.48, 0.16  # 0.01-grid fit, see --fit
    ours = 1 / (wb / 1.5278 + wa / 2.5624 + wm / 2.3483 + wo)
    lead = 1 / (wb / 1.6367 + wa / 3.032 + wm / 2.685 + wo)
    print("fitted composite ours", round(ours, 4), "with every track at board max", round(lead, 4))
    if "--show" in sys.argv:
        show(boards, load()[1])
