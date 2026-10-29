"""One fixed disjoint-boundary rule; CPU ownership/cost arithmetic, no GPU."""
from pathlib import Path
import bisect, hashlib, json, statistics, ast, re
ROOT = next((p for p in Path(__file__).resolve().parents if (p / 'STATE.json').exists()))
parent = ROOT / 'results/moe-unit-ownership-screen-20261029/screen.json'
raw = parent.read_bytes()
study = json.loads(raw)
parent_source = (ROOT / 'best/moe_verified.py').read_bytes()
assert hashlib.sha256(parent_source).hexdigest() == study['parent_moe_sha256'] == '331148716e8d1ba92b45cd29725be997d44b3a5235af74d75770bb9444100d50'
tree = ast.parse(parent_source)
cuda = next((n.value.value for n in tree.body if isinstance(n, ast.Assign) and any((isinstance(t, ast.Name) and t.id == 'CUDA_SRC' for t in n.targets))))
assert hashlib.sha256(cuda.encode()).hexdigest() == '1e60cf469ebff8d0f1e63a223f41f041374a1f30f90e4201a819ee178353270c'
TABLES = []
for name, macro in [('WGS_CA', 'OKC_WGSA'), ('WGS_CB', 'OKC_WGSB')]:
    assert f'constexpr int {name}[8] = {{{macro}}};' in cuda
    match = re.search(r'^#define ' + macro + r' ([^\r\n]+)', cuda, re.MULTILINE)
    assert match, name
    TABLES.append([int(x.strip()) for x in match.group(1).split('//', 1)[0].split(',')])
assert TABLES == [[112, 135, 140, 158, 173, 185, 200, 207], [100, 104, 115, 123, 132, 144, 156, 174]]

def model(jobs, grid, phase, allow_adjacent=False):
    table = TABLES[phase]
    units = (32, 16)[phase]
    stages = len(jobs) * units * 16
    cs = [table[(n + 7) // 8 - 1] for n in jobs]
    prefix = [0]
    for c in cs:
        prefix.append(prefix[-1] + c * units * 16)
    total = prefix[-1]
    assert total * grid < 2 ** 32

    def cost(s):
        if s == stages:
            return total
        j = s // (units * 16)
        return prefix[j] + s % (units * 16) * cs[j]

    def boundary(c):
        if not jobs:
            return 0
        if c == grid:
            return stages
        x = total * c // grid
        j = bisect.bisect_right(prefix, x) - 1
        return j * units * 16 + (x - prefix[j]) // cs[j]
    old = [boundary(c) for c in range(grid + 1)]
    new = old.copy()
    changed = []
    old_cost = [cost(b) - cost(a) for (a, b) in zip(old, old[1:])]
    for c in range(1, grid, 1 if allow_adjacent else 2):
        (a, x, b) = old[c - 1:c + 2]
        (left, right) = old_cost[c - 1:c + 1]
        y = x - 1 if x % 16 == 1 else x + 1 if x % 16 == 15 else x
        if y == x or not a <= y <= b:
            continue
        (nl, nr) = (cost(y) - cost(a), cost(b) - cost(y))
        if max(nl, nr) <= max(left, right):
            new[c] = y
            changed.append(c)
    new_cost = [cost(b) - cost(a) for (a, b) in zip(new, new[1:])]
    assert new[0] == 0 and new[-1] == stages and all((a <= b for (a, b) in zip(new, new[1:])))
    old_splits = {s // 16 for s in old[1:-1] if s % 16}
    new_splits = {s // 16 for s in new[1:-1] if s % 16}
    if not allow_adjacent:
        assert max(new_cost, default=0) <= max(old_cost, default=0)
        assert new_splits <= old_splits
        assert all((b - a >= 2 for (a, b) in zip(changed, changed[1:])))
    owner_changes = 0
    for s in range(0, stages, 16):
        xe = cost(s + 1)
        old_owner = min(grid - 1, (xe - 1) * grid // total)
        while old_owner + 1 < grid and total * (old_owner + 1) // grid < xe:
            old_owner += 1
        while old_owner > 0 and total * old_owner // grid >= xe:
            old_owner -= 1
        assert old[old_owner] <= s < old[old_owner + 1]
        exact = bisect.bisect_right(new, s) - 1
        if not allow_adjacent:
            repaired = old_owner + int(old_owner + 1 < grid and new[old_owner + 1] <= s)
            assert exact == repaired and new[repaired] <= s < new[repaired + 1]
            owner_changes += exact != old_owner

    def fixup_bytes(bounds, splits):
        atomic = 0
        read_zero = 0
        participants = 0
        for u in splits:
            lo = u * 16
            hi = lo + 16
            ntt = (jobs[u // units] + 7) // 8
            shares = sum((max(a, lo) < min(b, hi) for (a, b) in zip(bounds, bounds[1:])))
            assert shares >= 2
            participants += shares
            atomic += shares * 8192 * ntt
            read_zero += 2 * 8192 * ntt
        return dict(participants=participants, logical_atomic_payload_bytes=atomic, logical_finisher_read_zero_bytes=read_zero)
    (old_fix, new_fix) = (fixup_bytes(old, old_splits), fixup_bytes(new, new_splits))
    return (dict(changed_boundaries=len(changed), old_split_units=len(old_splits), new_split_units=len(new_splits), old_max_cost=max(old_cost, default=0), new_max_cost=max(new_cost, default=0), owner_changes=owner_changes, old_fixups=old_fix, new_fixups=new_fix), dict(jobs=jobs, grid=grid, phase=phase, old=old, new=new, old_cost=old_cost, new_cost=new_cost))
rows = []
owner_counterexample = None
adjacent_counterexample = None
checks = 0
for draw in study['rows']:
    for phase in (0, 1):
        (row, detail) = model(draw['jobs'], 132, phase)
        checks += 1
        rows.append(dict(m=draw['m'], distribution=draw['dist'], trial=draw['trial'], phase=phase, **row))
        if row['owner_changes'] and owner_counterexample is None:
            owner_counterexample = detail
        if adjacent_counterexample is None:
            (bad, d) = model(draw['jobs'], 132, phase, True)
            if bad['new_max_cost'] > bad['old_max_cost']:
                adjacent_counterexample = d
for jobs in [[]] + [[n] for n in range(1, 65)]:
    for grid in range(1, 161):
        for phase in (0, 1):
            model(jobs, grid, phase)
            checks += 1
for a in range(1, 65):
    for b in range(1, 65):
        for grid in (3, 32, 132, 160):
            for phase in (0, 1):
                model([a, b], grid, phase)
                checks += 1
for grid in (1, 32, 132, 160):
    for phase in (0, 1):
        model([64] * 136, grid, phase)
        checks += 1
summary = []
for m in (32, 128, 256):
    for dist in ('normal', 'skewed'):
        per = [r for r in rows if r['m'] == m and r['distribution'] == dist]
        summary.append(dict(m=m, distribution=dist, median_splits_removed=[statistics.median((r['old_split_units'] - r['new_split_units'] for r in per if r['phase'] == p)) for p in (0, 1)], median_participant_reduction=[statistics.median((1 - r['new_fixups']['participants'] / r['old_fixups']['participants'] for r in per if r['phase'] == p)) for p in (0, 1)]))
report = dict(input_sha256=hashlib.sha256(raw).hexdigest(), parent_moe_sha256=study['parent_moe_sha256'], method='Fixed odd cut indices; align only remainder1 backward or remainder15 forward when the affected pair maximum source-table stage cost does not increase', models_checked=checks, rows=rows, summary=summary, missing_owner_repair_counterexample=owner_counterexample, adjacent_cut_counterexample=adjacent_counterexample, critical_path_fraction_known=False, new_gpu_jobs=0, new_cuda_candidate=False, limitations=['Source-table cost is not latency', 'Logical fixup payload is not HBM traffic', 'Synthetic routing is not judged input', 'Added scheduler/inverse-owner instruction cost remains unmeasured'])
report['decision'] = 'Park this exact rule: small modeled fixup reduction, added scheduler/inverse-owner work unmeasured; no GPU run.'
out = ROOT / 'results/moe-disjoint-cut-screen-20261029'
out.mkdir(exist_ok=True)
raw_out = (json.dumps(report, indent=2) + '\n').encode()
path = out / 'screen.json'
assert not path.exists() or path.read_bytes() == raw_out, 'Frozen evidence collision'
path.write_bytes(raw_out)
print(json.dumps(dict(models_checked=checks, summary=summary, owner_counterexample=owner_counterexample is not None, adjacent_counterexample=adjacent_counterexample is not None, gpu_jobs=0)))
