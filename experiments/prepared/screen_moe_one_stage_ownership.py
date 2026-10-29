"""One fixed backward snap screen over saved synthetic draws; no GPU work."""
import bisect
import hashlib
import json
from pathlib import Path
import statistics

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'STATE.json').exists())


def main():
    path = ROOT / 'results/moe-unit-ownership-screen-20261029/screen.json'
    raw = path.read_bytes()
    parent = json.loads(raw)
    grid, rows, owner_checks = 132, [], 0
    for draw in parent['rows']:
        for phase, costs, units in [(0, [112, 135, 140, 158, 173, 185, 200, 207], 32),
                                    (1, [100, 104, 115, 123, 132, 144, 156, 174], 16)]:
            jobs = draw['jobs']
            cs = [costs[(n + 7) // 8 - 1] for n in jobs]
            prefix = [0]
            for cost in cs:
                prefix.append(prefix[-1] + cost * units * 16)
            total, stages = prefix[-1], len(jobs) * units * 16

            def boundary(c):
                if c == grid:
                    return stages
                pos = total * c // grid
                j = bisect.bisect_right(prefix, pos) - 1
                return j * units * 16 + (pos - prefix[j]) // cs[j]

            def cost_at(s):
                if s == stages:
                    return total
                j = s // (units * 16)
                return prefix[j] + s % (units * 16) * cs[j]

            old = [boundary(c) for c in range(grid + 1)]
            new = [s - 1 if s % 16 == 1 else s for s in old]
            assert new[0] == 0 and new[-1] == stages
            assert all(a <= b for a, b in zip(new, new[1:]))
            for s in range(0, stages, 16):
                j = s // (units * 16)
                # For unit-start s, snapped boundary<=s iff old boundary<=s+1.
                # The right edge of that old cost interval is the second stage.
                xe = prefix[j] + (s % (units * 16) + 2) * cs[j]
                owner = min(grid - 1, (xe * grid - 1) // total)
                assert new[owner] <= s < new[owner + 1]
                owner_checks += 1
            old_costs = [cost_at(b) - cost_at(a) for a, b in zip(old, old[1:])]
            new_costs = [cost_at(b) - cost_at(a) for a, b in zip(new, new[1:])]
            splits = lambda bounds: len({s // 16 for s in bounds[1:-1] if s % 16})
            rows.append(dict(m=draw['m'], dist=draw['dist'], trial=draw['trial'], phase=phase,
                             max_cost_ratio=max(new_costs) / max(old_costs),
                             splits_removed=splits(old) - splits(new), old_split_units=splits(old)))
    report = dict(method='One fixed backward snap at remainder1; CPU cost arithmetic, not a GPU candidate or timing',
                  parent_screen_sha256=hashlib.sha256(raw).hexdigest(), grid=grid,
                  owner_fixture_checks=owner_checks, rows=rows,
                  decision='Park pending phase evidence: few fixups removed and skewed-tail risk; no runtime change or GPU check justified by this model alone')
    folder = ROOT / 'results/moe-one-stage-ownership-screen-20261029'
    folder.mkdir(exist_ok=True)
    data = (json.dumps(report, indent=2) + '\n').encode()
    target = folder / 'screen.json'
    if target.exists():
        assert target.read_bytes() == data, 'Frozen evidence collision'
    else:
        target.write_bytes(data)
    for m in (32, 128, 256):
        for dist in ('normal', 'skewed'):
            chosen = [r for r in rows if r['m'] == m and r['dist'] == dist]
            print(json.dumps(dict(m=m, dist=dist, max_CTA_cost_ratio_medians=[statistics.median(
                r['max_cost_ratio'] for r in chosen if r['phase'] == p) for p in (0, 1)],
                removed_split_unit_medians=[statistics.median(r['splits_removed'] for r in chosen
                    if r['phase'] == p) for p in (0, 1)])))
    print(json.dumps(dict(cpu_owner_checks=owner_checks, gpu_jobs=0)))


if __name__ == '__main__':
    main()
