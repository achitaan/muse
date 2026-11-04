"""Reproducible profile ranking and single-weight-pass H200 roofline.

Reads only the explicitly supplied public first-check profile, not the
pending second check. No GPU, harness imports, network or OKC operations.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / 'candidates/profiles/dense_owner_epilogue.8ba14018820847fd.txt'
SHAPES = {'qkv_gate': (2304, 8192), 'o': (8192, 1024),
          'pooler': (4096, 8192), 'unpooler': (8192, 4096)}


def plan(layer, m):
    n, k = SHAPES[layer]
    pairs, kc = n // 128, k // 64
    stream = pairs < 64
    clusters = 66 if stream else pairs
    units = pairs * kc
    cuts = [units * g // clusters for g in range(clusters + 1)] if stream else [g * kc for g in range(clusters + 1)]
    owner = lambda u: ((u + 1) * clusters - 1) // units if stream else u // kc
    parts = [owner((p + 1) * kc - 1) - owner(p * kc) + 1 for p in range(pairs)]
    balance = units / (clusters * max(b - a for a, b in zip(cuts, cuts[1:])))
    assert sum(b - a for a, b in zip(cuts, cuts[1:])) == units
    assert min(parts) >= 1 and min(b - a for a, b in zip(cuts, cuts[1:])) > 0
    native = 16 if m <= 16 else 32 if m <= 32 else 128 if m <= 128 else 256
    stage = 8192 + native * 128
    smem = 1024 + 3 * stage + 64 + (native * 68 * 4 if stream else 0)
    return dict(n=n, k=k, m=m, native=native, stream=stream, clusters=clusters,
                maxparts=max(parts), parts=parts, balance=balance, smem_bytes=smem,
                sm_fraction=min(1., clusters * 2 / 132), weight_bytes=2 * n * k,
                activation_l2_bytes=native * k * 2 * pairs,
                fp32_partial_write_read_bytes=2 * sum(parts) * 2 * native * 64 * 4 if stream else 0)


def profile_rows(text):
    rows = []
    pattern = r'layer=(\w+) load=(decode|verify) (\d+) m=(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)x\s+([\d.]+)'
    for line in text.splitlines():
        hit = re.match(pattern, line)
        if hit:
            layer, kind, batch, m, sg, us, sp, sol = hit.groups()
            rows.append(dict(layer=layer, load=kind + batch, m=int(m), sglang_us=float(sg), ours_us=float(us),
                             speedup=float(sp), supplied_sol_us=float(sol)))
    assert len(rows) == 30
    totals = {load: sum(r['ours_us'] for r in rows if r['load'] == load) for load in dict.fromkeys(r['load'] for r in rows)}
    for row in rows:
        row['load_share'] = row['ours_us'] / totals[row['load']]
        row['equal_load_board_share'] = row['load_share'] / 6
        if row['layer'] in SHAPES:
            n, k = SHAPES[row['layer']]
            if row['layer'] == 'unpooler' and row['m'] <= 8:
                n = 1024
            m = row['m']
            row['raw_weight_once_us'] = (2 * n * k + 2 * m * k + 2 * m * n) / 4.8e6
            row['dense_compute_us'] = 2 * m * n * k / 989e6
            row['ideal_raw_floor_us'] = max(row['raw_weight_once_us'], row['dense_compute_us'])
    return rows, totals


def analyze():
    rows, totals = profile_rows(PROFILE.read_text())
    selected = [r for r in rows if r['layer'] in SHAPES and (r['m'] in (128, 256) or r['layer'] == 'o' and r['m'] in (16, 32))]
    models = []
    for name, eff, fixed, partial in [('ideal', 1., 0., 0.), ('design', .90, 1., .75),
                                       ('lower_efficiency', .80, 1., .75), ('higher_latency', .90, 1.5, 1.)]:
        updated, cells = dict(totals), []
        for row in selected:
            p = plan(row['layer'], row['m'])
            bw = row['raw_weight_once_us'] / eff
            compute = row['dense_compute_us'] / (eff * p['sm_fraction'] * p['balance'])
            target = max(bw, compute) + fixed + (partial if p['stream'] else 0.)
            ratio = target / row['ours_us']
            adopt = ratio <= .975
            updated[row['load']] -= max(0., row['ours_us'] - target) if adopt else 0.
            cells.append(dict(layer=row['layer'], m=row['m'], ours_us=row['ours_us'],
                              raw_floor_us=row['ideal_raw_floor_us'], target_us=target, ratio=ratio,
                              modeled_adopt=adopt, plan=p))
        gain = math.exp(sum(math.log(totals[load] / updated[load]) for load in totals) / 6)
        models.append(dict(name=name, efficiency=eff, fixed_us=fixed, reduction_us=partial,
                           cells=cells, new_load_totals_us=updated, board_gain=gain))
    return dict(profile=str(PROFILE), rows=rows, load_totals_us=totals,
                lowest_speedups=sorted(rows, key=lambda r: r['speedup'])[:12],
                highest_time_shares=sorted(rows, key=lambda r: -r['equal_load_board_share'])[:12], models=models,
                build_gate_passed=models[0]['board_gain'] >= 1.03 and models[1]['board_gain'] >= 1.03)


if __name__ == '__main__':
    result = analyze()
    (ROOT / 'results/dense-redesign-roofline.json').write_text(json.dumps(result, indent=2) + '\n')
    print('LOAD TOTALS', result['load_totals_us'])
    for model in result['models']:
        print(model['name'], 'board_gain=%.6f' % model['board_gain'])
        if model['name'] == 'design':
            for cell in model['cells']:
                print(cell['layer'], cell['m'], '%.2f -> %.3f' % (cell['ours_us'], cell['target_us']),
                      'adopt', cell['modeled_adopt'])
    print('BUILD GATE', result['build_gate_passed'])
