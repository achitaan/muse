"""Synthetic CPU cost screen of whole-unit ownership; not GPU performance evidence."""
import bisect
import hashlib
import json
from pathlib import Path
import platform
import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'STATE.json').exists())
CA = [112, 135, 140, 158, 173, 185, 200, 207]
CB = [100, 104, 115, 123, 132, 144, 156, 174]


def main():
    rng = np.random.default_rng(20261029)
    grid, rows = 132, []
    for m in (32, 128, 256):
        for dist in ('normal', 'skewed'):
            for trial in range(32):
                scores = rng.random((m, 256))
                if dist == 'normal':
                    scores += .03 * rng.standard_normal(256) / np.sqrt(-np.log(np.maximum(rng.random(256), 1e-12)))
                else:
                    scores += rng.random(256) ** 6 * 4
                ids = np.argpartition(scores, -8, axis=1)[:, -8:]
                counts = np.bincount(ids.ravel(), minlength=256).reshape(8, 32)
                local = counts[np.argmax(counts.sum(1))]
                jobs = [min(64, int(n) - start) for n in local if n > 8 for start in range(0, int(n), 64)]
                jobs += [int(n) for n in local if 0 < n <= 8]
                record = dict(m=m, dist=dist, trial=trial, local_routes=int(local.sum()),
                              expert_counts=local.tolist(), jobs=jobs, phases=[])
                for phase, costs, units in ((0, CA, 32), (1, CB, 16)):
                    cs = [costs[(n + 7) // 8 - 1] for n in jobs]
                    prefix = [0]
                    for cost in cs:
                        prefix.append(prefix[-1] + cost * units * 16)
                    total = prefix[-1]

                    def boundary(c):
                        if c == grid:
                            return len(jobs) * units * 16
                        x = total * c // grid
                        j = bisect.bisect_right(prefix, x) - 1
                        return j * units * 16 + (x - prefix[j]) // cs[j]

                    def cost_at(stage):
                        if stage == len(jobs) * units * 16:
                            return total
                        j = stage // (units * 16)
                        return prefix[j] + stage % (units * 16) * cs[j]

                    old = [boundary(c) for c in range(grid + 1)]
                    aligned = [s // 16 * 16 for s in old]
                    old_costs = [cost_at(b) - cost_at(a) for a, b in zip(old, old[1:])]
                    new_costs = [cost_at(b) - cost_at(a) for a, b in zip(aligned, aligned[1:])]
                    assert old[0] == aligned[0] == 0 and old[-1] == aligned[-1]
                    assert all(a <= b for a, b in zip(aligned, aligned[1:]))
                    record['phases'].append(dict(phase=phase, total_units=len(jobs) * units,
                        split_units=len({s // 16 for s in old[1:-1] if s % 16}),
                        aligned_max_cost_over_old=max(new_costs) / max(old_costs),
                        aligned_idle_ctas=sum(c == 0 for c in new_costs)))
                rows.append(record)
    report = dict(seed=20261029, grid=grid, numpy=np.__version__, python=platform.python_version(),
                  parent_moe_sha256=hashlib.sha256((ROOT / 'best/moe_verified.py').read_bytes()).hexdigest(),
                  method='NumPy public-formula-inspired stress draws; not official inputs, measured traffic, GPU timings or predictions',
                  rows=rows, decision='Do not spend a GPU check on unconditional whole-unit alignment: modeled tail work rises substantially')
    out = ROOT / 'results/moe-unit-ownership-screen-20261029'
    out.mkdir(exist_ok=True)
    data = (json.dumps(report, indent=2) + '\n').encode()
    target = out / 'screen.json'
    if target.exists():
        assert target.read_bytes() == data, 'Frozen screen differs; retain the old environment/version record'
    else:
        target.write_bytes(data)
    for m in (32, 128, 256):
        for dist in ('normal', 'skewed'):
            selected = [r for r in rows if r['m'] == m and r['dist'] == dist]
            print(json.dumps(dict(m=m, dist=dist, median_phase_ratios=[float(np.median([
                r['phases'][p]['aligned_max_cost_over_old'] for r in selected])) for p in (0, 1)])))


if __name__ == '__main__':
    main()
