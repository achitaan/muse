"""Compare a Decode check against the retained D9 parent job, per shape and per load.

usage: python scripts/compare_decode_runs.py <result.json> [gates.json]
Affected shapes: sliding layers and batch-1 full layers (prefetch enabled);
controls: full layers with batch > 1. Ratios are parent/candidate median_ns.
"""
import json
import math
from pathlib import Path
import sys

LANE = Path(__file__).resolve().parents[1]
PARENT = LANE / 'results/decoder-segment-repair-20261028/profiles/submission.9560c915e85f4f50.json'


def benchmarks(obj):
    if 'benchmarks' in obj:
        return obj['benchmarks']
    body = obj.get('result') or obj
    return [r for r in body.get('records', []) if r.get('event') == 'benchmark']


def profiles(obj):
    if 'profiles' in obj:
        return obj['profiles']
    body = obj.get('result') or obj
    return [r for r in body.get('records', []) if r.get('event') == 'profile']


def key(b):
    c = b['case']
    return (c['load'], c['layer'])


def gm(values):
    values = list(values)
    return math.exp(sum(math.log(v) for v in values) / len(values)) if values else float('nan')


def main():
    cand = json.loads(Path(sys.argv[1]).read_bytes())
    gates = json.loads(Path(sys.argv[2]).read_bytes()) if len(sys.argv) > 2 else None
    base = {key(b): b for b in benchmarks(json.loads(PARENT.read_bytes()))}
    new = {key(b): b for b in benchmarks(cand)}
    body = cand.get('result') or {}
    rows, affected, control = [], [], []
    loads = {}
    for k in sorted(base, key=lambda k: (k[0], k[1])):
        if k not in new:
            continue
        (b, n) = (base[k], new[k])
        ratio = b['median_ns'] / n['median_ns']
        is_aff = k[1] == 'sliding' or b['case']['batch'] == 1
        (affected if is_aff else control).append(ratio)
        rows.append(dict(load=k[0], layer=k[1], tokens=b['case']['batch'] * b['case'].get('q_len', 1),
                         parent_us=round(b['median_ns'] / 1e3, 2), cand_us=round(n['median_ns'] / 1e3, 2),
                         ratio=round(ratio, 4), cand_speedup=round(n['speedup'], 4), parent_speedup=round(b['speedup'], 4),
                         group='affected' if is_aff else 'control'))
        w = b['case']['weight']
        agg = loads.setdefault(k[0], [0.0, 0.0, 0.0])
        agg[0] += w * b['baseline_ns']
        agg[1] += w * b['median_ns']
        agg[2] += w * n['median_ns']
    for r in rows:
        print(f"{r['load']:10s} {r['layer']:7s} T={r['tokens']:3d} parent {r['parent_us']:8.2f} cand {r['cand_us']:8.2f} ratio {r['ratio']:.4f} [{r['group']}]")
    per_load = {l: dict(parent=a[0] / a[1], cand=a[0] / a[2]) for (l, a) in loads.items()}
    for (l, v) in per_load.items():
        print(f"load {l:10s} parent {v['parent']:.4f} cand-vs-parent-baseline {v['cand']:.4f} gain {v['cand'] / v['parent']:.4f}")
    summary = dict(affected_gm=gm(affected), control_gm=gm(control), n_affected=len(affected), n_control=len(control),
                   track_speedup=body.get('speedup'), speedup_ci=body.get('speedup_ci'), passed=body.get('passed'),
                   tests=sum(1 for r in body.get('records', []) if r.get('event') == 'test'),
                   reconstructed_parent_track=gm(v['parent'] for v in per_load.values()),
                   reconstructed_cand_track_on_parent_baselines=gm(v['cand'] for v in per_load.values()))
    log = body.get('log_tail') or ''
    summary['l2pf_log'] = [l for l in log.splitlines() if '[l2pf]' in l][-6:]
    pk = []
    for p in profiles(cand):
        rep = p.get('report') or ''
        hit = [l.strip()[:120] for l in rep.splitlines() if '_pfk' in l]
        pk.append(dict(load=p['case']['load'], layer=p['case']['layer'], prefetch_kernel=hit[:1]))
    summary['profile_prefetch_kernel'] = pk
    if gates:
        g = gates['gates']
        summary['gate_affected'] = summary['affected_gm'] >= g['affected_gm_ratio_min']
        summary['gate_control'] = g['control_gm_ratio_range'][0] <= summary['control_gm'] <= g['control_gm_ratio_range'][1]
        summary['gate_track'] = (summary['track_speedup'] or 0) >= g['track_speedup_min']
    print(json.dumps(summary, indent=2))
    out = Path(sys.argv[1]).with_name('comparison.json')
    out.write_text(json.dumps(dict(rows=rows, per_load=per_load, summary=summary), indent=2))


if __name__ == '__main__':
    main()
