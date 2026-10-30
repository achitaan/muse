"""Bounded helper-local norm aggregation screen; never imports CUDA runtime."""
import ast
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/decode-moe-plan-20261030/norm-register'


def source():
    raw = (ROOT / 'best/moe_verified.py').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == '331148716e8d1ba92b45cd29725be997d44b3a5235af74d75770bb9444100d50'
    vals = {n.targets[0].id: ast.literal_eval(n.value) for n in ast.parse(raw).body
            if isinstance(n, ast.Assign) and getattr(n.targets[0], 'id', '')
            in ('CUDA_SRC', 'TINY_DEFINES', 'TINY2_DEFINES')}
    parent = vals['CUDA_SRC']
    a = parent.index('__device__ void ws_helper(')
    b = parent.index('__device__ __forceinline__ void skinny_split(', a)
    helper = parent[a:b]
    edits = []

    def edit(old, new):
        nonlocal helper
        assert helper.count(old) == 1, old[:100]
        helper = helper.replace(old, new, 1)
        edits.append((old, new))

    edit('  for (int q = 0; q < Nh; ++q) {', '''  int norm_job = -1;
  float norm_sum0 = 0.f, norm_sum1 = 0.f;
  for (int q = 0; q < Nh; ++q) {''')
    flush = '''      if (norm_job >= 0) {
        const int old_st = S.jS[norm_job], old_cnt = S.jC[norm_job];
        if (2 * tt < old_cnt) atomicAdd(&ss_cur()[old_st + 2 * tt], norm_sum0);
        if (2 * tt + 1 < old_cnt) atomicAdd(&ss_cur()[old_st + 2 * tt + 1], norm_sum1);
      }'''
    edit('''    } else {
      const float gsc = p.gsB[ce] * TWO62;''', '''    } else {
      if (gg == 0 && j != norm_job) {
''' + flush + '''
        norm_job = j;
        norm_sum0 = 0.f; norm_sum1 = 0.f;
      }
      const float gsc = p.gsB[ce] * TWO62;''')
    edit('if (gg == 0 && tk < cnt) atomicAdd(&ss_cur()[st + tk], sq);', '''if (gg == 0 && tk < cnt) {
          const float nsq = (__float_as_uint(sq) & 0x7f800000u) == 0 ? 0.f : sq;
          if (h) norm_sum1 += nsq; else norm_sum0 += nsq;
        }''')
    edit('''  __threadfence();
  WS_JOIN();  // every epilogue store of this CTA is done before the grid barrier''', '''  if (gg == 0) {
''' + flush + '''
  }
  __threadfence();
  WS_JOIN();  // every epilogue store of this CTA is done before the grid barrier''')
    inverse = helper
    for old, new in reversed(edits):
        assert inverse.count(new) == 1
        inverse = inverse.replace(new, old, 1)
    assert inverse == parent[a:b]
    src = parent[:a] + helper + parent[b:]
    src = src.replace('struct Smem {', 'static_assert(OKC_TINY && OKC_WS && !SKDYN && SK_T == 8, "helper norm requires static tiny WS");\nstruct Smem {', 1)
    src = src.replace('namespace okc', 'namespace norm_reg').replace('okc::', 'norm_reg::').replace('moe_fused', 'moe_reg_fused')
    return src, vals


def ownership():
    cases = 0
    for jobs in range(33):
        counts = [1 + j % 8 for j in range(jobs)]
        for grid in (1, 2, 7, 132, 148, 512):
            for tail in (1, 2):
                covered = [0] * (256 * jobs)
                for c in range(grid):
                    b0, b1 = 256 * jobs * c // grid, 256 * jobs * (c + 1) // grid
                    stop = b1 - min(tail, b1 - b0)
                    # Flush transitions use the old job and its own count.
                    active, pending, published = -1, [], []
                    for v in range(b0, stop):
                        j = v // 256
                        if j != active:
                            if active >= 0:
                                published.extend((active, tk, unit) for unit in pending for tk in range(counts[active]))
                            active, pending = j, []
                        pending.append(v)
                    if active >= 0:
                        published.extend((active, tk, unit) for unit in pending for tk in range(counts[active]))
                    expected = [(v // 256, tk, v) for v in range(b0, stop) for tk in range(counts[v // 256])]
                    assert sorted(published) == sorted(expected)
                    for v in range(b0, stop): covered[v] += 1
                    for v in range(stop, b1): covered[v] += 1
                assert all(n == 1 for n in covered)
                cases += 1
    # Reproduce just routing metadata for the existing timing fixtures.
    fixtures = []
    for m in range(1, 9):
        for pattern in ('one', 'skew'):
            rng = random.Random(20261030 + 1009 * 7 + 31 * m)
            hot = rng.sample(list(range(32)), 8)
            counts = [0] * 32
            for t in range(m):
                n = 1 if pattern == 'one' else 1 + t % 3
                for e in rng.sample(hot, n): counts[e] += 1
                rng.sample(list(range(32, 256)), 8 - n)
            counts = [n for n in counts if n]
            assert max(counts) <= 8
            old = 256 * sum(counts)
            new = 0
            tail = 2 if m <= 4 else 1
            for c in range(132):
                b0, b1 = 256 * len(counts) * c // 132, 256 * len(counts) * (c + 1) // 132
                stop = b1 - min(tail, b1 - b0)
                new += sum(counts[v // 256] for v in range(stop, b1))
                new += sum(counts[j] for j in set(v // 256 for v in range(b0, stop)))
            fixtures.append(dict(M=m, pattern=pattern, jobs=len(counts), parent_norm_atomics=old,
                                 regrouped_norm_atomics=new, reduction_fraction=1-new/old))
    return dict(cases=cases, exactly_once=True, fixtures=fixtures,
                limitation='Integer contribution/flush model, not CUDA races, floating-point error or elapsed time.')


def main():
    OUT.mkdir(exist_ok=True)
    src, vals = source()
    (OUT / 'candidate.cu').write_bytes(src.encode())
    sys.path.insert(0, str(ROOT.parent / 'opus-fresh/experiments/opus_fresh'))
    import nvrtc_screen_alternatives as nv
    result = dict(source_sha256=hashlib.sha256(src.encode()).hexdigest(), exact_inverse=True,
                  ownership=ownership(), GPU_calls=0, admitted_for_GPU=False, compiler={})
    for name in ('TINY', 'TINY2'):
        for prmt in (False, True):
            label = name + ('_PRMT' if prmt else '_LUT')
            flags = list(vals[name + '_DEFINES']) + ['-DOKC_DQ7=1', '-DOKC_NS=3', '-DOKC_NSF=3']
            if prmt: flags += ['-DOKC_PRMTSC=1']
            r = nv.NVRTC().compile(src.split('#include <torch/extension.h>', 1)[0].encode(),
                                  ['&norm_reg::moe_reg_fused<false>'], flags, tags=True, lineinfo=False)
            (OUT / (label + '.log')).write_bytes(r['log'].encode())
            assert r['rc'] == 0, r['log'][-3000:]
            result['compiler'][label] = nv.parse_ptxas(r['log'])
            print(label, json.dumps(result['compiler'][label]), flush=True)
    result['zero_stack_spills'] = all(k['stack_frame'] == k['spill_stores'] == k['spill_loads'] == 0
                                      for ks in result['compiler'].values() for k in ks)
    (OUT / 'cpu-audit.json').write_bytes((json.dumps(result, indent=2) + '\n').encode())


if __name__ == '__main__':
    main()
