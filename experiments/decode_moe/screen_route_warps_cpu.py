"""Fixed tiny-route ownership screen. CPU metadata model and NVRTC only; no GPU calls."""
import ast
import hashlib
import json
from pathlib import Path
import random
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/decode-moe-plan-20261030/route-warps'
BASE = '331148716e8d1ba92b45cd29725be997d44b3a5235af74d75770bb9444100d50'


def derive():
    raw = (ROOT / 'best/moe_verified.py').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == BASE
    vals = {n.targets[0].id: ast.literal_eval(n.value) for n in ast.parse(raw).body
            if isinstance(n, ast.Assign) and getattr(n.targets[0], 'id', '')
            in ('CUDA_SRC', 'TINY_DEFINES', 'TINY2_DEFINES')}
    parent = vals['CUDA_SRC']
    a = parent.index('__device__ void route(')
    b = parent.index('// ---------------------------------------------------------------- epilogues', a)
    original = parent[a:b]
    route = original
    edits = []

    def edit(old, new):
        nonlocal route
        assert route.count(old) == 1, old
        route = route.replace(old, new, 1)
        edits.append((old, new))

    edit('  const int E = p.M * 8;', '''  const int E = p.M * 8;
#if OKC_ROUTE_WARPS
  static_assert(OKC_TINY && NW == 8 && (OKC_ROUTE_WARPS == 1 || OKC_ROUTE_WARPS == 2), "tiny route only");
  constexpr int RW = OKC_ROUTE_WARPS;
#else
  constexpr int RW = NW;
#endif''')
    edit('if (tid < NW * 32) S.wc', 'if (tid < RW * 32) S.wc')
    old = 'for (int w = 0; w < NW; ++w)'
    assert route.count(old) == 2
    route = route.replace(old, 'for (int w = 0; w < RW; ++w)')
    edits.append((old, 'for (int w = 0; w < RW; ++w)'))
    # For E <= RW*32 <= NT, CH=32. Each warp initializes and reads its own eid/wc row.
    edit('''  CTA_SYNC();
  for (int base = i0; base < i1; base += 32) {''', '''#if OKC_ROUTE_WARPS
  __syncwarp();
#else
  CTA_SYNC();
#endif
  for (int base = i0; base < i1; base += 32) {''')
    edit('''  CTA_SYNC();
  if (warp == 0) {''', '''#if OKC_ROUTE_WARPS == 1
  __syncwarp();
#else
  CTA_SYNC();
#endif
  if (warp == 0) {''')
    inverse = route
    for old, new in reversed(edits):
        inverse = inverse.replace(new, old)
    assert inverse == original
    candidate = parent[:a] + route + parent[b:]
    header = '#ifndef OKC_ROUTE_WARPS\n#define OKC_ROUTE_WARPS 0\n#endif\n'
    candidate = header + candidate
    guard = '''
#if OKC_ROUTE_WARPS
  TORCH_CHECK(p.M > 0 && p.M <= 4 * OKC_ROUTE_WARPS && p.nloc > 0 && p.nloc <= 32,
              "tiny route capacity exceeded");
#endif'''
    point = '  p.M = (int)x.size(0); p.lo = (int)lo; p.nloc = (int)nloc;'
    assert candidate.count(point) == 1
    candidate = candidate.replace(point, point + guard)
    assert candidate.removeprefix(header).replace(route, original).replace(point + guard, point) == parent
    # No other reader can observe the omitted scratch rows.
    assert all(not re.search(r'\bS\.wc\b', x) for x in (parent[:a], parent[b:]))
    return parent, candidate, vals


def model(ids, lo, nloc, rw):
    E = len(ids)
    assert E <= rw * 32 <= 256
    # Poison unused wc rows; every read must come from a written row.
    wc = [[None] * 32 for _ in range(8)]
    eid = [e - lo if 0 <= e - lo < nloc else -1 for e in ids]
    for w in range(rw):
        wc[w] = [0] * 32
        for e in eid[w * 32:(w + 1) * 32]:
            if e >= 0:
                wc[w][e] += 1
    counts = [sum(wc[w][e] for w in range(rw)) if e < nloc else 0 for e in range(32)]
    off, carry = [], 0
    for n in counts:
        off.append(carry)
        carry += n
    ent = [None] * carry
    pos = [-1] * E
    for e in range(32):
        start = off[e]
        for w in range(rw):
            peers = [i for i in range(w * 32, min(E, (w + 1) * 32)) if eid[i] == e]
            for rank, i in enumerate(peers):
                ent[start + rank] = i
                pos[i] = start + rank
            start += wc[w][e]
    # Job partition/order is untouched; exact counts/offsets/ent/pos imply the same jobs.
    return eid, counts, off, ent, pos


def ownership():
    rng = random.Random(2026103103)
    cases = 0
    for m in range(1, 9):
        rw = 1 if m <= 4 else 2
        for lo in (0, 32, 224):
            for nloc in (1, 2, 7, 16, 32):
                E = m * 8
                patterns = [[lo - 1] * E, [lo] * E,
                            [lo + i % nloc for i in range(E)],
                            [lo + (i % (nloc + 2)) - 1 for i in range(E)]]
                patterns += [[rng.randrange(-1, 257) for _ in range(E)] for _ in range(20)]
                for ids in patterns:
                    result = model(ids, lo, nloc, rw)
                    assert result == model(ids, lo, nloc, 8)
                    eid, counts, off, ent, pos = result
                    expected = sorted((i for i, e in enumerate(eid) if e >= 0), key=lambda i: (eid[i], i))
                    assert ent == expected
                    assert all(p == -1 if eid[i] < 0 else ent[p] == i for i, p in enumerate(pos))
                    cases += 1
    # Static access/visibility proof: own-warp writes before barrier1; only RW1 removes barrier2.
    for E in range(8, 65, 8):
        for i in range(E):
            assert (i // 32) == ((i % 256) // 32)
    return dict(cases=cases, duplicate_ids_included=True, unused_rows_poisoned=True,
                all_count_offset_entry_position_metadata_exact=True,
                limitation='CPU integer model and source access proof; not GPU race, numerical or timing validation.')


def main():
    OUT.mkdir(exist_ok=True)
    (OUT / '.gitignore').write_bytes(b'*.cubin\n')
    parent, candidate, vals = derive()
    (OUT / 'candidate.cu').write_bytes(candidate.encode())
    sys.path.insert(0, str(ROOT.parent / 'opus-fresh/experiments/opus_fresh'))
    import nvrtc_screen_alternatives as nv

    class Capture(nv.NVRTC):
        def output(self, program, kind):
            value = super().output(program, kind)
            if kind == 'CUBIN':
                self.cubin = value
            return value

    report = dict(baseline_sha256=BASE, baseline_CUDA_sha256=hashlib.sha256(parent.encode()).hexdigest(),
                  candidate_CUDA_sha256=hashlib.sha256(candidate.encode()).hexdigest(), exact_inverse=True,
                  metadata=ownership(), resources={}, GPU_jobs=0, admitted_for_GPU=False)
    exe = ROOT.parents[2] / '.tools/cuda-compiler-13.0/nvidia/cu13/bin/cuobjdump.exe'
    for family, rw in [('TINY', 1), ('TINY2', 2)]:
        for prmt in (True, False):
            key = family + ('_PRMT' if prmt else '_LUT')
            report['resources'][key] = {}
            for label, source in [('parent', parent), ('candidate', candidate)]:
                flags = list(vals[family + '_DEFINES']) + ['-DOKC_DQ7=1', '-DOKC_NS=3', '-DOKC_NSF=3']
                if prmt:
                    flags += ['-DOKC_PRMTSC=1']
                if label == 'candidate':
                    flags += ['-DOKC_ROUTE_WARPS=' + str(rw)]
                compiler = Capture()
                result = compiler.compile(source.split('#include <torch/extension.h>', 1)[0].encode(),
                                          ['&okc::moe_fused<false>'], flags, tags=True, lineinfo=False)
                stem = key + '-' + label
                (OUT / (stem + '.log')).write_bytes(result['log'].encode())
                assert result['rc'] == 0, result['log'][-3000:]
                binary = OUT / (stem + '.cubin')
                binary.write_bytes(compiler.cubin)
                sass = subprocess.check_output([str(exe), '-sass', str(binary)], text=True)
                (OUT / (stem + '.sass')).write_bytes(sass.encode())
                report['resources'][key][label] = dict(kernels=nv.parse_ptxas(result['log']),
                    nvrtc_warning_messages=re.findall(r'warning #[^\n]+', result['log']),
                    instructions=len(re.findall(r'/\*[0-9a-fA-F]+\*/\s+.*?;', sass)),
                    local_instructions=len(re.findall(r'\b(?:LDL|STL)\b', sass)),
                    CTA_barrier_instructions=len(re.findall(r'\bBAR\.SYNC\b', sass)))
                print(key, label, json.dumps(report['resources'][key][label]), flush=True)
    clean = True
    for pair in report['resources'].values():
        a, b = pair['parent'], pair['candidate']
        assert a['nvrtc_warning_messages'] == b['nvrtc_warning_messages']
        clean &= b['local_instructions'] == 0 and len(a['kernels']) == len(b['kernels'])
        for x, y in zip(a['kernels'], b['kernels']):
            clean &= all(y[k] <= x[k] for k in ('registers', 'stack_frame', 'spill_stores', 'spill_loads'))
    report.update(no_resource_regression=clean,
                  route_CTA_barriers={'parent':4, 'TINY_candidate':2, 'TINY2_candidate':3},
                  route_histogram_rows={'parent':8, 'TINY_candidate':1, 'TINY2_candidate':2},
                  decision='CPU resource screen passed;GPU numerical/timing and affected-load opportunity remain unproven' if clean else 'CPU resource reject;no GPU')
    (OUT / 'cpu-audit.json').write_bytes((json.dumps(report, indent=2) + '\n').encode())
    print(json.dumps({k:v for k,v in report.items() if k != 'resources'}, indent=2))


if __name__ == '__main__':
    main()
