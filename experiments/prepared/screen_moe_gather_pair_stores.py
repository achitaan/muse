"""CPU ownership/sector screen; no runtime edit, GPU measurement or submission."""
import ast
import bisect
import hashlib
import json
from pathlib import Path
import statistics

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'STATE.json').exists())
BASE = '331148716e8d1ba92b45cd29725be997d44b3a5235af74d75770bb9444100d50'


def function_text(source, name):
    start = source.index('__device__ void ' + name + '(')
    brace = source.index('{', start)
    depth = 1
    end = brace + 1
    while depth:
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    return source[start:end]


def address(job_start, cnt, row, tid):
    """Byte offset: exact retained uint4 destination expression."""
    ntt = (cnt + 7) // 8
    return job_start * 4096 * 2 + (((tid >> 5) * ntt + (row >> 3)) * 32
                                  + (tid & 31)) * 128 + (row & 7) * 16


def sectors(addresses):
    return {byte // 32 for a in addresses for byte in range(a, a + 16)}


def transpose(a, b, wrong_shuffle=False):
    # Swap register row bit with lane bit 4. All 32 lanes participate.
    incoming = [b[lane] if lane < 16 else a[lane] for lane in range(32)]
    cross = [incoming[lane ^ (8 if wrong_shuffle else 16)] for lane in range(32)]
    out0 = [a[lane] if lane < 16 else cross[lane] for lane in range(32)]
    out1 = [cross[lane] if lane < 16 else b[lane] for lane in range(32)]
    return out0, out1


def pair_proof(cnt, row, warp, wrong_shuffle=False):
    # Unique four-word payloads expose row, lane, component and byte ownership.
    a = [tuple((row << 24) | (warp << 20) | (lane << 8) | k for k in range(4))
         for lane in range(32)]
    b = [tuple(((row + 1) << 24) | (warp << 20) | (lane << 8) | k for k in range(4))
         for lane in range(32)]
    before = {address(8, cnt, row + rr, warp * 32 + lane): values[lane]
              for rr, values in enumerate((a, b)) for lane in range(32)}
    outputs = transpose(a, b, wrong_shuffle)
    after = {}
    requests = []
    for q, values in enumerate(outputs):
        addrs = []
        for lane in range(32):
            source_lane = (lane & 15) + 16 * q
            source_row = row + (lane >> 4)
            dest = address(8, cnt, source_row, warp * 32 + source_lane)
            assert dest not in after
            after[dest] = values[lane]
            addrs.append(dest)
        requests.append(len(sectors(addrs)))
    assert before == after, 'Payload/ownership changed'
    assert requests == [16, 16]
    assert [len(sectors([address(8, cnt, row + rr, warp * 32 + lane)
                         for lane in range(32)])) for rr in (0, 1)] == [32, 32]
    return len(after)


def count_pairs(jobs, grid):
    # Preserve each CTA's range and the existing eight-row load batch.
    prefix = [0]
    for n in jobs:
        prefix.append(prefix[-1] + ((n + 7) // 8) * 8)
    pairs = 0
    valid = sum(jobs)
    for cta in range(grid):
        lo, hi = prefix[-1] * cta // grid, prefix[-1] * (cta + 1) // grid
        for batch in range(lo, hi, 8):
            q = batch
            stop = min(batch + 8, hi)
            while q < stop:
                j = bisect.bisect_right(prefix, q) - 1
                r = q - prefix[j]
                if r % 2 == 0 and r + 1 < jobs[j] and q + 1 < stop:
                    assert q + 1 < prefix[j + 1]
                    pairs += 1
                    q += 2
                else:
                    q += 1
    assert 2 * pairs <= valid
    return pairs, valid


def main():
    raw = (ROOT / 'best/moe_verified.py').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == BASE
    tree = ast.parse(raw)
    cuda = next(n.value for n in ast.walk(tree) if isinstance(n, ast.Constant)
                and isinstance(n.value, str) and '__device__ void wgs_gather(' in n.value)
    gather = function_text(cuda, 'wgs_gather')
    assert '#define OKC_GATHB 8' in cuda
    assert 'if (d[k]) *d[k] = v[k];' in gather
    assert 'static_assert(NT * 8 == HD' in gather
    assert 'red.release.gpu.global.add.s32' in gather
    ownership = cases = 0
    for cnt in range(1, 65):
        for row in range(0, cnt - 1, 2):
            for warp in range(16):
                ownership += pair_proof(cnt, row, warp)
                cases += 1
    try:
        pair_proof(8, 0, 0, wrong_shuffle=True)
    except AssertionError:
        mutant_detected = True
    else:
        raise AssertionError('Wrong lane-exchange mutant escaped')
    # Explicitly cover no rows, odd CTA boundaries, padding and fat-job splits.
    edge_cases = [([], 132), ([1], 132), ([1], 1), ([2], 1), ([7], 1),
                  ([8], 3), ([9], 7), ([64, 1], 132), ([64] * 48, 160)]
    edges = [dict(jobs=j, grid=g, pairs=count_pairs(j, g)[0]) for j, g in edge_cases]
    parent_path = ROOT / 'results/moe-unit-ownership-screen-20261029/screen.json'
    parent_raw = parent_path.read_bytes()
    parent = json.loads(parent_raw)
    rows = []
    for draw in parent['rows']:
        pairs, valid = count_pairs(draw['jobs'], 132)
        old = valid * 512
        new = old - pairs * 512
        rows.append(dict(m=draw['m'], dist=draw['dist'], trial=draw['trial'],
                         routes=valid, pairs=pairs,
                         paired_route_fraction=2 * pairs / valid if valid else 0,
                         old_store_sector_requests=old, new_store_sector_requests=new,
                         store_request_ratio=new / old if old else 1,
                         unchanged_payload_bytes=valid * 8192,
                         fewer_requested_sector_bytes=(old - new) * 32))
    summaries = []
    for m in (32, 128, 256):
        for dist in ('normal', 'skewed'):
            selected = [r for r in rows if r['m'] == m and r['dist'] == dist]
            summaries.append(dict(m=m, dist=dist, median_paired_route_fraction=statistics.median(
                r['paired_route_fraction'] for r in selected),
                median_store_request_ratio=statistics.median(r['store_request_ratio'] for r in selected),
                median_fewer_requested_sector_bytes=statistics.median(
                    r['fewer_requested_sector_bytes'] for r in selected)))
    report = dict(cpu_only=True, gpu_jobs=0, runtime_candidate_written=False,
                  baseline_sha256=BASE, cuda_sha256=hashlib.sha256(cuda.encode()).hexdigest(),
                  gather_sha256=hashlib.sha256(gather.encode()).hexdigest(),
                  parent_screen_sha256=hashlib.sha256(parent_raw).hexdigest(),
                  pair_layout_cases=cases, exact_uint4_ownership_checks=ownership,
                  wrong_shuffle_mutant_detected=mutant_detected, edge_cases=edges,
                  mechanism='Exchange lane bit4 and the row register bit for two valid even/odd rows within the same n8 block; unchanged loads, stores, payload, CTA ranges and readiness protocol',
                  sector_model='Per 16B warp store instruction, count distinct aligned32B sectors. Assumes compiled stores remain16B; no claim about L1/L2 merging, measured HBM traffic or GPU nanoseconds.',
                  source_cost='Four 32-bit warp shuffles per row pair, conditional register moves and address/predicate work; extra liveness and exact SASS are unmeasured.',
                  fixtures='Reuse192 synthetic stress draws, selecting their busiest32-expert shard; not official routing data or a representative-distribution estimate',
                  numerical='Opaque uint4 payload exchange only; no floating point operations or changed reduction order in this proposal',
                  decision='Offline ownership screen passes. Park GPU evaluation until exposed gather/store cost and likely whole-layer benefit justify added shuffle/register work.',
                  source='https://docs.nvidia.com/cuda/archive/13.0.0/cuda-c-best-practices-guide/index.html#coalesced-access-to-global-memory',
                  summaries=summaries, rows=rows)
    folder = ROOT / 'results/moe-gather-pair-store-screen-20261029'
    folder.mkdir(exist_ok=True)
    data = (json.dumps(report, indent=2) + '\n').encode()
    target = folder / 'screen.json'
    if target.exists():
        assert target.read_bytes() == data, 'Frozen evidence collision'
    else:
        target.write_bytes(data)
    print(json.dumps({k: report[k] for k in ('cpu_only', 'gpu_jobs', 'pair_layout_cases',
                                            'exact_uint4_ownership_checks', 'summaries', 'decision')}))


if __name__ == '__main__':
    main()
