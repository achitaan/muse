"""Conditional score arithmetic only; no GPU timings or profiler fractions."""
import hashlib
import json
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'STATE.json').exists())


def main():
    path = ROOT / 'results/retained-load-opportunities-20261029/loads.json'
    raw = path.read_bytes()
    retained = json.loads(raw)
    changed = ['decode 32', 'verify 16', 'verify 32']
    scenarios = []
    for gain in (1.01, 1.03, 1.08):
        # Six equally weighted loads; only these three improve by the same
        # multiplicative factor, while all controls remain exactly unchanged.
        load_gain = gain ** 2
        fraction_removed = 1 - 1 / load_gain
        rows = []
        for name in changed:
            load = retained['studies']['decode']['loads'][name]
            weight = sum(row['weight'] for row in load['cases'])
            assert weight == 62
            saved_ns = load['retained_load_ns'] * fraction_removed
            rows.append(dict(load=name, tokens=load['cases'][0]['tokens'],
                             weighted_load_saving_ms=saved_ns / 1e6,
                             weighted_average_layer_saving_us=saved_ns / weight / 1000))
        scenarios.append(dict(track_gain=gain, equal_changed_load_gain=load_gain,
                              changed_load_time_reduction=fraction_removed, required_savings=rows))
    moe_only_m32 = dict(track_gain=1.015, affected_loads=1,
                       required_load_gain=1.015 ** 6,
                       required_time_reduction=1 - 1.015 ** -6)
    report = dict(input_sha256=hashlib.sha256(raw).hexdigest(),
                  method='Conditional dilution identities using historical Decode benchmark load times; no forecast',
                  changed_loads=changed, unchanged_loads=['decode 1', 'decode 16', 'verify 1'],
                  equal_multiplicative_load_gains_assumed=True, scenarios=scenarios,
                  standalone_one_load_gate=moe_only_m32,
                  critical_path_fraction_used=False, overall_board_prediction=False,
                  new_gpu_jobs=0)
    lines = ['# Conditional Decode improvement budget', '',
             'This is arithmetic from retained D9 benchmark load times, not a speed forecast or an overall-board prediction. It uses no profiler duration as a critical-path fraction.', '',
             'Assume only M32, M128 and M256 improve, by the same multiplicative load factor, with the other three loads unchanged. A desired track gain G then requires each changed load to speed up by G squared, removing `1 - 1/G^2` of its old weighted time.', '',
             '| Desired Decode track gain | Each changed load time removed | M32 average layer saving us | M128 average layer saving us | M256 average layer saving us |',
             '| ---: | ---: | ---: | ---: | ---: |']
    for row in scenarios:
        values = [x['weighted_average_layer_saving_us'] for x in row['required_savings']]
        lines.append(f"| {(row['track_gain']-1)*100:.0f}% | {row['changed_load_time_reduction']*100:.2f}% | {values[0]:.2f} | {values[1]:.2f} | {values[2]:.2f} |")
    lines += ['', 'Average layer saving is `(46*delta_sliding + 16*delta_full)/62`. It does not require both layer types to improve equally. It charges every added operation, launch, store, reduction and lost overlap to the new complete layer time.', '',
              'For a standalone MoE change affecting only M32, a1.5% track gain requires about9.34% speedup on that one load, or8.55% less load time. A3% per-load speedup on just one load gives only about0.49% track gain. These dilution identities constrain experiment value; they are not acceptance evidence.', '',
              'Do not convert a synthetic phase span into any required saving above. A candidate must demonstrate the complete weighted layer change and unchanged controls, preserving whole-layer correctness. The diagnostic is intended to choose where to investigate, not to satisfy this performance gate.', '',
              'Reproduce: `python experiments/prepared/screen_required_layer_savings.py`. Exact input hash and full-precision arithmetic are in `budget.json`.', '']
    folder = ROOT / 'results/required-layer-savings-20261029'
    folder.mkdir(exist_ok=True)
    for name, data in [('budget.json', (json.dumps(report, indent=2) + '\n').encode()),
                       ('review.md', '\n'.join(lines).encode())]:
        target = folder / name
        if target.exists():
            assert target.read_bytes() == data, ('Frozen evidence collision', name)
        else:
            target.write_bytes(data)
    print(json.dumps(dict(report=str(folder.relative_to(ROOT)), scenarios=scenarios, gpu_jobs=0)))


if __name__ == '__main__':
    main()
