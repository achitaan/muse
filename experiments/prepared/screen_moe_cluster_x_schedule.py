"""Paired-output-unit cluster schedule model; no GPU execution."""
import bisect, hashlib, json, statistics, ast, re
from pathlib import Path
ROOT = next((p for p in Path(__file__).resolve().parents if (p / 'STATE.json').exists()))
INPUT = ROOT / 'results/moe-unit-ownership-screen-20261029/screen.json'
raw = INPUT.read_bytes()
parent = json.loads(raw)
TABLES = ([112, 135, 140, 158, 173, 185, 200, 207], [100, 104, 115, 123, 132, 144, 156, 174])
verified = (ROOT / 'best/moe_verified.py').read_bytes()
assert hashlib.sha256(verified).hexdigest() == parent['parent_moe_sha256'] == '331148716e8d1ba92b45cd29725be997d44b3a5235af74d75770bb9444100d50'
tree = ast.parse(verified)
cuda = next((n.value.value for n in tree.body if isinstance(n, ast.Assign) and any((isinstance(t, ast.Name) and t.id == 'CUDA_SRC' for t in n.targets))))
assert hashlib.sha256(cuda.encode()).hexdigest() == '1e60cf469ebff8d0f1e63a223f41f041374a1f30f90e4201a819ee178353270c'
for (macro, expected) in zip(['OKC_WGSA', 'OKC_WGSB'], TABLES):
    match = re.search('^#define ' + macro + ' ([^\\r\\n]+)', cuda, re.MULTILINE)
    assert match and [int(v.strip()) for v in match.group(1).split('//', 1)[0].split(',')] == expected

def layout(jobs, phase, grid, cluster):
    unit_count = (32, 16)[phase] // cluster
    table = TABLES[phase]
    costs = [table[(x + 7) // 8 - 1] for x in jobs]
    prefix = [0]
    for cost in costs:
        prefix.append(prefix[-1] + cost * unit_count * 16)
    total = prefix[-1]
    stages = len(jobs) * unit_count * 16
    groups = grid // cluster
    assert grid % cluster == 0 and total * groups < 2 ** 32

    def at(s):
        if s == stages:
            return total
        j = s // (unit_count * 16)
        return prefix[j] + s % (unit_count * 16) * costs[j]
    cuts = []
    for c in range(groups + 1):
        if not jobs:
            s = 0
        elif c == groups:
            s = stages
        else:
            target = total * c // groups
            j = bisect.bisect_right(prefix, target) - 1
            s = j * unit_count * 16 + (target - prefix[j]) // costs[j]
        cuts.append(s)
    assert all((a <= b for (a, b) in zip(cuts, cuts[1:])))
    work = [at(b) - at(a) for (a, b) in zip(cuts, cuts[1:]) for rank in range(cluster)]
    coverage = bytearray(stages * cluster)
    for (c, (a, b)) in enumerate(zip(cuts, cuts[1:])):
        for s in range(a, b):
            for rank in range(cluster):
                actual = (s // 16 * cluster + rank) * 16 + s % 16
                assert coverage[actual] == 0
                coverage[actual] = 1
    assert all((x == 1 for x in coverage))
    for actual_unit in range(len(jobs) * (32, 16)[phase]):
        unit_group = actual_unit // cluster
        rank = actual_unit % cluster
        s = unit_group * 16
        xe = at(s + 1)
        owner = min(groups - 1, (xe - 1) * groups // total)
        while owner + 1 < groups and total * (owner + 1) // groups < xe:
            owner += 1
        while owner > 0 and total * owner // groups >= xe:
            owner -= 1
        assert owner == bisect.bisect_right(cuts, s) - 1
        actual_owner = cluster * owner + rank
        assert actual_owner < grid and cuts[actual_owner // cluster] <= s < cuts[actual_owner // cluster + 1]
    splits = {s // 16 for s in cuts[1:-1] if s % 16}
    partial_shares = 0
    for unit in splits:
        partial_shares += sum((max(a, unit * 16) < min(b, unit * 16 + 16) for (a, b) in zip(cuts, cuts[1:]))) * cluster
    source_bytes = sum((unit_count * 16 * ((n + 7) // 8 * 8) * 512 for n in jobs))
    return (dict(max_CTA_cost=max(work, default=0), total_CTA_cost=sum(work), partial_units=len(splits) * cluster, partial_shares=partial_shares, source_copy_requests=stages, logical_source_copy_bytes=source_bytes, logical_destination_copy_bytes=source_bytes * cluster), cuts)

def compare(jobs, phase, grid):
    (old, old_cuts) = layout(jobs, phase, grid, 1)
    (new, new_cuts) = layout(jobs, phase, grid, 2)
    assert old['total_CTA_cost'] == new['total_CTA_cost']
    assert new['source_copy_requests'] * 2 == old['source_copy_requests']
    assert new['logical_source_copy_bytes'] * 2 == old['logical_source_copy_bytes']
    assert new['logical_destination_copy_bytes'] == old['logical_destination_copy_bytes']
    return dict(jobs=jobs, phase=phase, grid=grid, old=old, new=new, max_cost_ratio=new['max_CTA_cost'] / old['max_CTA_cost'] if old['max_CTA_cost'] else 1)
rows = []
fixtures = 0
for draw in parent['rows']:
    for phase in (0, 1):
        rows.append(dict(m=draw['m'], distribution=draw['dist'], trial=draw['trial'], **compare(draw['jobs'], phase, 132)))
        fixtures += 1
for n in range(1, 65):
    for grid in (2, 16, 32, 66, 128, 132, 160):
        for phase in (0, 1):
            compare([n], phase, grid)
            fixtures += 1
for m in (1, 4, 8, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 191, 192, 255, 256, 257, 384, 385):
    for distribution in ('remote', 'onehot', 'uniform8'):
        if distribution == 'remote':
            counts = [0] * 32
        elif distribution == 'onehot':
            counts = [m] + [0] * 31
        else:
            counts = [8 * m // 32 + int(e < 8 * m % 32) for e in range(32)]
        jobs = [min(64, n - i) for n in counts if n > 8 for i in range(0, n, 64)] + [n for n in counts if 0 < n <= 8]
        for grid in (2, 132, 160):
            for phase in (0, 1):
                compare(jobs, phase, grid)
                fixtures += 1
summary = []
for m in (32, 128, 256):
    for distribution in ('normal', 'skewed'):
        selected = [r for r in rows if r['m'] == m and r['distribution'] == distribution]
        summary.append(dict(m=m, distribution=distribution, max_cost_ratio_median=[statistics.median((r['max_cost_ratio'] for r in selected if r['phase'] == p)) for p in (0, 1)], max_cost_ratio_worst=[max((r['max_cost_ratio'] for r in selected if r['phase'] == p)) for p in (0, 1)], old_source_MiB_medians=[statistics.median((r['old']['logical_source_copy_bytes'] for r in selected if r['phase'] == p)) / 2 ** 20 for p in (0, 1)], partial_share_ratio_median=[statistics.median((r['new']['partial_shares'] / r['old']['partial_shares'] for r in selected if r['phase'] == p)) for p in (0, 1)]))
report = dict(input_sha256=hashlib.sha256(raw).hexdigest(), parent_submission_sha256=parent['parent_moe_sha256'], method='One fixed cluster2: pair consecutive output units, preserve stream-K within paired-unit groups, balance over G/2 clusters', models_checked=fixtures, rows=rows, summary=summary, unit_stage_bijection_valid=True, owner_slot_inverse_valid=True, grid_barrier_CTA_count_unchanged=True, source_copy_requests_halved=True, destination_bytes_unchanged=True, HBM_traffic_measured=False, latency_measured=False, barrier_protocol_proven=False, actual_resources_known=False, GPU_jobs=0, CUDA_candidate_implemented=False)
out = ROOT / 'results/moe-cluster-x-feasibility-20261029'
out.mkdir(exist_ok=True)
path = out / 'schedule-model.json'
data = (json.dumps(report, indent=2) + '\n').encode()
assert not path.exists() or path.read_bytes() == data, 'Frozen evidence collision'
path.write_bytes(data)
print(json.dumps(dict(models_checked=fixtures, summary=summary, GPU_jobs=0)))
