import json
import math
import sys
from collections import defaultdict

r = json.load(open(sys.argv[1]))['result']
rows = [x for x in r['records'] if x.get('event') == 'timing' or ('speedup' in x and 'case' in x and 'baseline_ns' in x)]
loads = defaultdict(lambda: [0.0, 0.0])
print(f"{'load':10} {'layer':9} {'m':>4} {'ours_us':>8} {'sglang_us':>9} {'sol_us':>7} {'x_sol':>6} {'spd':>6}")
for x in rows:
    c = x['case']
    ours = x['baseline_ns'] / x['speedup'] / 1000
    base = x['baseline_ns'] / 1000
    sol = x['speed_of_light_ns'] / 1000
    loads[c['load']][0] += base
    loads[c['load']][1] += ours
    print(f"{c['load']:10} {c['layer']:9} {c['m']:>4} {ours:8.2f} {base:9.2f} {sol:7.2f} {ours / sol:6.2f} {x['speedup']:6.3f}")
g = []
for (k, (b, o)) in loads.items():
    g.append(b / o)
    print(f'load {k:10} sglang {b:7.2f} ours {o:7.2f} spd {b / o:.4f}  1us_gain {(b / (o - 1) / (b / o) - 1) * 100 / len(loads):.3f}%')
print('geomean', math.exp(sum(map(math.log, g)) / len(g)), 'reported', r['speedup'])
