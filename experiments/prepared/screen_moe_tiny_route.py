"""CPU-only equivalence screen for a future one-warp M1..4 route prologue.

No CUDA, GPU jobs, API calls, submission changes or timing claims.
"""
from pathlib import Path
import ast
import hashlib
import itertools
import json
import random
import re

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'STATE.json').exists())
PARENT = '331148716e8d1ba92b45cd29725be997d44b3a5235af74d75770bb9444100d50'
raw = (ROOT / 'best/moe_verified.py').read_bytes()
assert hashlib.sha256(raw).hexdigest() == PARENT
tree = ast.parse(raw)
cuda = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == 'CUDA_SRC' for t in n.targets))
route_start = cuda.index('__device__ void route(const Params& p, Smem& S) {')
route_end = cuda.index('// ---------------------------------------------------------------- epilogues', route_start)
route = cuda[route_start:route_end]
# No consumer outside route may depend on its per-warp count workspace.
without_route = cuda[:route_start] + cuda[route_end:]
assert not re.search(r'\bS\.wc\s*\[', without_route)
assert route.count('CTA_SYNC();') == 4
sk = re.search(r'#define OKC_SKT (\d+)', cuda)
ft = re.search(r'#define OKC_FT_T (\d+)', cuda)
assert sk and ft
SK, FT = int(sk[1]), int(ft[1])


def finish(counts, offsets, ent, pos):
    fat, skinny = [], []
    for expert, count in enumerate(counts):
        if count > SK:
            for q in range((count + FT - 1) // FT):
                fat.append((expert, offsets[expert] + q * FT, min(FT, count - q * FT)))
        elif count:
            skinny.append((expert, offsets[expert], count))
    order = sorted(range(len(fat)), key=lambda j: (-((fat[j][2] + 7) // 8), j))
    return dict(counts=counts, offsets=offsets, ent=ent, pos=pos,
                fat=fat, skinny=skinny, fat_order=order)


def reference(ids, lo, count):
    groups = [[i for i, e in enumerate(ids) if e == lo + expert] for expert in range(32)]
    groups[count:] = [[] for _ in range(32 - count)]
    counts = list(map(len, groups))
    ent, offsets, pos = [], [], [-1] * len(ids)
    for group in groups:
        offsets.append(len(ent))
        for i in group:
            pos[i] = len(ent)
            ent.append(i)
    return finish(counts, offsets, ent, pos)


def warp_model(ids, lo, count):
    # Lane i owns entry i; only one leader writes each expert's count.
    assert 0 < len(ids) <= 32
    eid = [e - lo if 0 <= e - lo < count else -1 for e in ids]
    peer_masks = [sum(1 << j for j, v in enumerate(eid) if v == e) for e in eid]
    counts = [0] * 32
    for lane, e in enumerate(eid):
        mask = peer_masks[lane]
        if e >= 0 and lane == (mask & -mask).bit_length() - 1:
            counts[e] = mask.bit_count()
    # Warp inclusive prefix: each round reads the preceding round's values.
    inclusive = counts[:]
    for d in (1, 2, 4, 8, 16):
        old = inclusive[:]
        for lane in range(d, 32):
            inclusive[lane] += old[lane - d]
    offsets = [a - b for a, b in zip(inclusive, counts)]
    ent, pos = [-999] * sum(counts), [-1] * len(ids)
    for lane, e in enumerate(eid):
        if e >= 0:
            q = offsets[e] + (peer_masks[lane] & ((1 << lane) - 1)).bit_count()
            assert ent[q] == -999
            ent[q], pos[lane] = lane, q
    assert -999 not in ent
    return finish(counts, offsets, ent, pos)


cases = 0
for ids in itertools.product((-1, 0, 1, 31), repeat=8):
    assert reference(ids, 0, 32) == warp_model(ids, 0, 32)
    cases += 1
rng = random.Random(20261029)
seen_counts = set()
for m in range(1, 5):
    for _ in range(5000):
        lo = rng.randrange(8) * 32
        count = rng.randrange(1, 33)
        ids = [rng.randrange(256) for _ in range(m * 8)]
        if rng.random() < .4:
            hot = lo + rng.randrange(count)
            ids = [hot if rng.random() < .75 else e for e in ids]
        a, b = reference(ids, lo, count), warp_model(ids, lo, count)
        assert a == b
        seen_counts.update(a['counts'])
        cases += 1
    for nlocal in range(m * 8 + 1):
        ids = [0] * nlocal + [255] * (m * 8 - nlocal)
        assert reference(ids, 0, 32) == warp_model(ids, 0, 32)
        seen_counts.add(nlocal)
        cases += 1
assert seen_counts == set(range(33))
report = dict(parent_sha256=PARENT, cpu_only=True, cases=cases,
              exhaustive_m1_four_symbol_patterns=4**8, random_cases=20000,
              all_duplicate_counts_0_through_32=True,
              stable_expert_entries_positions_jobs_and_fat_order_equal=True,
              original_route_CTA_barriers=4,
              per_warp_count_workspace_unread_outside_route=True,
              cuda_implementation_written=False, gpu_verified=False,
              expected_speedup=None,
              decision='Park pending exposed tiny-routing cost and resource model; do not spend a GPU check on this CPU screen')
output = ROOT / '.tools/moe-tiny-route-cpu-screen.json'
output.write_bytes((json.dumps(report, indent=2) + '\n').encode())
print(json.dumps(report, indent=2))
