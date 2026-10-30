"""CPU route ownership model and optional SM90 compilation, without GPU use."""

import argparse
import hashlib
import json
import random
import sys

from tinyroute_source import derive, RESULTS, BASE_SHA, HERE


def generic(ids, lo, nloc):
    eid = [e - lo if 0 <= e - lo < nloc else -1 for e in ids]
    cnt = [eid.count(e) for e in range(32)]
    off = [sum(cnt[:e]) for e in range(32)]
    ent = [i for e in range(32) for i, v in enumerate(eid) if v == e]
    pos = [-1] * len(ids)
    for q, i in enumerate(ent):
        pos[i] = q
    jobs = [(e, off[e], c) for e, c in enumerate(cnt) if c]
    return dict(eid=eid, cnt=cnt, off=off, ent=ent, pos=pos, jobs=jobs)


def bitplanes(ids, lo, nloc):
    # This is an instruction-level integer model of the new branch. The
    # independent oracle above is stable counting sort, not this construction.
    E = len(ids)
    es = [e - lo if 0 <= e - lo < nloc else -1 for e in ids] + [-1] * (64 - E)
    masks = []
    for base in (0, 32):
        valid = sum(1 << lane for lane in range(32) if es[base + lane] >= 0)
        planes = [sum(1 << lane for lane in range(32) if es[base + lane] & (1 << b)) for b in range(5)]
        counts = []
        for expert in range(32):
            own = valid
            for b, plane in enumerate(planes):
                own &= plane if expert & (1 << b) else ~plane
            counts.append(own.bit_count())
        masks.append(counts)
    cnt = [a + b for a, b in zip(*masks)]
    if max(cnt) > 8:
        return None
    incl = cnt[:]
    for d in (1, 2, 4, 8, 16):
        old = incl[:]
        incl = [v + old[lane - d] if lane >= d else v for lane, v in enumerate(old)]
    off = [v - c for v, c in zip(incl, cnt)]
    present = sum(1 << e for e, c in enumerate(cnt) if c)
    jobs = [None] * present.bit_count()
    ent = [None] * sum(cnt)
    pos = [-1] * E
    for lane in range(32):
        before = (1 << lane) - 1
        if cnt[lane]:
            jobs[(present & before).bit_count()] = (lane, off[lane], cnt[lane])
        for base in (0, 32):
            i = base + lane
            e = es[i]
            peers = sum(1 << l for l in range(32) if es[base + l] == e)
            if i < E and e >= 0:
                q = off[e] + (masks[0][e] if base else 0) + (peers & before).bit_count()
                assert ent[q] is None
                ent[q] = i
                pos[i] = q
    return dict(eid=es[:E], cnt=cnt, off=off, ent=ent, pos=pos, jobs=jobs)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--compile', action='store_true'); args = p.parse_args()
    raw, parent, modified, flags = derive()
    rng = random.Random(20261030)
    checked = rejected = 0
    for M in range(1, 9):
        for lo in (0, 31, 128, 224):
            for nloc in (1, 13, 32):
                cases = [[lo] * (8 * M), [-1] * (8 * M), [lo + nloc] * (8 * M)]
                cases += [sum([rng.sample(range(256), 8) for _ in range(M)], []) for _ in range(25)]
                cases += [[rng.choice([lo, lo + nloc - 1, -1, 256]) for _ in range(8 * M)] for _ in range(10)]
                for ids in cases:
                    candidate = bitplanes(ids, lo, nloc)
                    if candidate is None:
                        assert max(generic(ids, lo, nloc)['cnt']) > 8
                        rejected += 1
                    else:
                        assert candidate == generic(ids, lo, nloc), (M, lo, nloc, ids)
                        checked += 1
    result = dict(baseline_sha256=BASE_SHA, parent_CUDA_sha256=hashlib.sha256(parent.encode()).hexdigest(),
                  candidate_CUDA_sha256=hashlib.sha256(modified.encode()).hexdigest(), exact_inverse=True,
                  seed=20261030, exact_route_tables=checked, oversized_histogram_fallbacks=rejected,
                  GPU_validated=False)
    if args.compile:
        sys.path.insert(0, str(HERE.parents[2] / 'opus-fresh/experiments/opus_fresh'))
        import nvrtc_screen_alternatives as nv
        compiler = nv.NVRTC()
        result['compiles'] = {}
        for name in ['TINY', 'TINY2']:
            for prmt in [False, True]:
                for arm, source in [('parent', parent), ('candidate', modified)]:
                    defs = list(flags[name]) + ['-DOKC_DQ7=1', '-DOKC_NS=3', '-DOKC_NSF=3']
                    if prmt: defs += ['-DOKC_PRMTSC=1']
                    if arm == 'candidate': defs += ['-DOKC_ROUTE_WARP=1']
                    key = name + ('_PRMT_' if prmt else '_LUT_') + arm
                    kernel = 'moe_tr8_fused' if arm == 'candidate' else 'moe_fused'
                    r = compiler.compile(source.split('#include <torch/extension.h>', 1)[0].encode(),
                        ['&okc::' + kernel + '<false>'], defs, tags=True, lineinfo=False)
                    (RESULTS / (key + '.log')).write_text(r['log'])
                    result['compiles'][key] = dict(rc=r['rc'], kernels=nv.parse_ptxas(r['log']) if r['rc'] == 0 else [])
                    print(key, result['compiles'][key], flush=True)
                    assert r['rc'] == 0, r['log'][-2000:]
    (RESULTS / 'cpu-evidence.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
