"""Per-shape comparison of a check against a parent run, with drift normalized by control shapes.

usage: python scripts/compare_runs.py <parent.json> <result.json> <gates.json>
gates.json: {"affected": [[load, layer], ...], "per_shape_win": 1.03, ...}
Ratios are parent/candidate median_ns; control_gm is the geometric mean over shapes not listed
as affected; a shape's normalized ratio is ratio / control_gm.
"""
import json
import math
from pathlib import Path
import sys


def benchmarks(obj):
    if 'benchmarks' in obj:
        return obj['benchmarks']
    body = obj.get('result') or obj
    return [r for r in body.get('records', []) if r.get('event') == 'benchmark']


def key(b):
    c = b['case']
    return (c['load'], str(c.get('layer', '')))


def gm(v):
    v = list(v)
    return math.exp(sum(math.log(x) for x in v) / len(v)) if v else float('nan')


def main():
    parent = json.loads(Path(sys.argv[1]).read_bytes())
    cand = json.loads(Path(sys.argv[2]).read_bytes())
    gates = json.loads(Path(sys.argv[3]).read_bytes())
    affected = {tuple(a) for a in gates['affected']}
    base = {key(b): b for b in benchmarks(parent)}
    new = {key(b): b for b in benchmarks(cand)}
    ratios = {k: base[k]['median_ns'] / new[k]['median_ns'] for k in base if k in new}
    control = gm(v for (k, v) in ratios.items() if k not in affected)
    rows = []
    for k in sorted(ratios):
        r = ratios[k]
        rows.append(dict(load=k[0], layer=k[1], parent_us=round(base[k]['median_ns'] / 1e3, 2),
                         cand_us=round(new[k]['median_ns'] / 1e3, 2), ratio=round(r, 4),
                         normalized=round(r / control, 4), affected=k in affected,
                         win=(k in affected and r / control >= gates.get('per_shape_win', 1.03))))
        print(f"{k[0]:10s} {k[1]:8s} parent {rows[-1]['parent_us']:8.2f} cand {rows[-1]['cand_us']:8.2f} "
              f"ratio {r:.4f} norm {r / control:.4f} {'AFFECTED' if k in affected else 'control'}{' WIN' if rows[-1]['win'] else ''}")
    body = cand.get('result') or {}
    summary = dict(control_gm=control, affected_gm=gm(v for (k, v) in ratios.items() if k in affected),
                   affected_gm_normalized=gm(v / control for (k, v) in ratios.items() if k in affected),
                   track_speedup=body.get('speedup'), speedup_ci=body.get('speedup_ci'), passed=body.get('passed'),
                   tests=sum(1 for r in body.get('records', []) if r.get('event') == 'test'),
                   winners=[[r['load'], r['layer']] for r in rows if r['win']])
    print(json.dumps(summary, indent=2))
    Path(sys.argv[2]).with_name('comparison.json').write_text(json.dumps(dict(rows=rows, summary=summary), indent=2))


if __name__ == '__main__':
    main()
