"""One fixed router-only CTA ownership experiment; CPU checks and compilation only."""
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/decode-moe-plan-20261030/router-sidecar'
DECODER = 'c15df9d3d83c55f4ba64be810e56502563e6cfe22bb5ef22172a1feb287654e5'


def derive():
    raw = (ROOT / 'experiments/decode_moe/decoder_pdli.py').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == DECODER
    tree = ast.parse(raw)
    node = next(n for n in tree.body if isinstance(n, ast.Assign)
                and getattr(n.targets[0], 'id', '') == '_PDLI_EXPECTED')
    expected = ast.literal_eval(node.value)
    parent = (ROOT / 'results/decode-moe-plan-20261030/pdli-cpu/r1.cu').read_bytes()
    assert hashlib.sha256(parent).hexdigest() == expected['r1']
    parent = parent.decode()
    a = parent.index('__device__ __forceinline__ void v18s_r2_router(')
    b = parent.index('// R2f: the extra block', a)
    original = parent[a:b]
    helper = original
    changes = []

    def h(old, new):
        nonlocal helper
        assert helper.count(old) == 1
        helper = helper.replace(old, new, 1)
        changes.append((old, new))

    h('const int blk = blockIdx.x, nb = blk & 15, ks = blk >> 4, row = nb * 16 + warp;',
      'const int blk = blockIdx.x - 129, nb = blk & 7, ks = blk >> 3, row = nb * 32 + warp;')
    h('  if (warp >= 16) return;\n', '')
    h('bar.sync 1, 512;', 'bar.sync 1, 1024;')
    h('atomicAdd(ra.counter, 1u);', 'atomicAdd(ra.counter, 2u);')
    assert helper[helper.index('  float a[8];'):helper.index('  if (lane == 0) __threadfence();')] == original[
        original.index('  float a[8];'):original.index('  if (lane == 0) __threadfence();')]
    inverse = helper
    for old, new in reversed(changes):
        if new:
            assert inverse.count(new) == 1
            inverse = inverse.replace(new, old, 1)
        else:
            inverse = inverse.replace('  float a[8];', old + '  float a[8];', 1)
    assert inverse == original
    source = parent[:a] + helper + parent[b:]
    edits = [(original, helper)]

    def edit(old, new):
        nonlocal source
        assert source.count(old) == 1, old[:80]
        source = source.replace(old, new, 1)
        edits.append((old, new))

    anchor = '      // PDL: router input loads follow the independent weight preamble.'
    edit(anchor, '''      static_assert(CWR == 2 && CWK == 16 && CS == 1 && CN == 4096 && CK == 8192 && MT == 1 && RT == 1 && RED == 0,
                    "fixed router sidecar geometry");
      if (blockIdx.x >= 129) {
        asm volatile("griddepcontrol.wait;" ::: "memory");
        v18s_r2_router(ra, warp, lane);
        return;
      }
      // Pooler CTAs retain their independent preamble and dependency wait.''')
    edit('    v18s_r2_router(ra, warp, lane);\n    if constexpr',
         '    // Router units have separate CTAs; the pooler arithmetic is unchanged.\n    if constexpr')
    edit('lc.gridDim = dim3(N / RB + ((OPT & 64) ? 1 : 0), S, 1);',
         'lc.gridDim = dim3(N / RB + ((OPT & 64) ? 65 : 0), S, 1);')
    inverse = source
    for old, new in reversed(edits):
        assert inverse.count(new) == 1
        inverse = inverse.replace(new, old, 1)
    assert inverse == parent
    topk = lambda s: s[s.index('__device__ __forceinline__ void v18s_r2_topk('):s.index('template <int FMT', s.index('__device__ __forceinline__ void v18s_r2_topk('))]
    assert topk(source) == topk(parent)
    return parent, source


def ownership():
    old = {(b >> 4, (b & 15) * 16 + w): (b, w) for b in range(128) for w in range(16)}
    new = {(b >> 3, (b & 7) * 32 + w): (b, w) for b in range(64) for w in range(32)}
    assert len(old) == len(new) == 2048 and old.keys() == new.keys()
    checks = 0
    for (ks, row), (b, w) in new.items():
        old_b, old_w = old[ks, row]
        assert old_b == 2 * b + w // 16 and old_w == w % 16
        assert old_b >> 4 == b >> 3
        for m in (0, 1, 3, 7):
            assert ((ks * 8 + m) * 256 + row) == (((old_b >> 4) * 8 + m) * 256 + (old_b & 15) * 16 + old_w)
            for lane in (0, 1, 15, 31):
                for j in (0, 7, 16, 31):
                    assert row * 8192 + ks * 1024 + j * 32 + lane == ((old_b & 15) * 16 + old_w) * 8192 + (old_b >> 4) * 1024 + j * 32 + lane
                    checks += 1
    assert 64 * 2 == 128
    return dict(partial_outputs_exactly_once=2048, address_checks=checks,
                original_FMA_and_shuffle_subtree_exact=True, original_topk_subtree_exact=True,
                arrival_total=128, arrival_increment=2, CTA_count=193, pooler_CTA_count=128,
                router_CTA_count=64, topk_CTA_count=1,
                per_CTA_threads=1024, original_register_budget=64,
                progress_argument='Only top-k waits on this grid;pooler and router CTAs do not. With capacity for at least two CTAs, no resident-grid circular wait is introduced. Incoming-PDL ancestor progress remains a separate required GPU proof.',
                scheduling_assumption='No correctness argument assumes CUDA schedules blocks in index order.',
                limitation='Integer ownership/address proof,not GPU numerical,replay,race,occupancy or latency evidence.')


def main():
    OUT.mkdir(exist_ok=True)
    (OUT / '.gitignore').write_bytes(b'*.cubin\n')
    parent, candidate = derive()
    (OUT / 'candidate.cu').write_bytes(candidate.encode())
    from check_r2_cpu import device_part
    from check_pdl_diag_cpu import Capture, MAIN
    import nvrtc_screen_alternatives as nv
    report = dict(decoder_sha256=DECODER, parent_CUDA_sha256=hashlib.sha256(parent.encode()).hexdigest(),
                  candidate_CUDA_sha256=hashlib.sha256(candidate.encode()).hexdigest(), exact_inverse=True,
                  ownership=ownership(), GPU_jobs=0, admitted_for_GPU=False, resources={})
    symbol = '&v18s_gemv<1,1,1,1,0,0,1024,0,4096,8192,2,16,8,1,70>'
    for label, source in [('parent', parent), ('candidate', candidate)]:
        compiler = Capture()
        r = compiler.compile(device_part(source).encode(), [symbol], [], tags=True, lineinfo=True)
        (OUT / (label + '.log')).write_bytes(r['log'].encode())
        assert r['rc'] == 0, r['log'][-4000:]
        binary = OUT / (label + '.cubin')
        binary.write_bytes(compiler.cubin)
        sass = subprocess.check_output([str(MAIN / '.tools/cuda-compiler-13.0/nvidia/cu13/bin/cuobjdump.exe'), '-sass', str(binary)], text=True)
        (OUT / (label + '.sass')).write_bytes(sass.encode())
        report['resources'][label] = dict(kernels=nv.parse_ptxas(r['log']),
            local_instructions=len(re.findall(r'\b(?:LDL|STL)\b', sass)),
            dependency_wait_instructions=len(re.findall(r'\bACQBULK\b', sass)),
            outgoing_trigger_instructions=len(re.findall(r'\bPREEXIT\b', sass)),
            cubin_sha256=r['cubin_sha256'])
        print(label, json.dumps(report['resources'][label]), flush=True)
    parent_res, new_res = report['resources']['parent'], report['resources']['candidate']
    clean = new_res['local_instructions'] == 0 and all(k['registers'] <= 64 and k['stack_frame'] == k['spill_stores'] == k['spill_loads'] == 0
                                                       for k in new_res['kernels'])
    report.update(clean_resource_screen=clean,
                  decision='CPU screen passed;source CFG/PDL and scheduling cost evidence required before GPU admission' if clean else 'CPU reject:1024-thread resource budget/spill regression;no GPU')
    (OUT / 'cpu-audit.json').write_bytes((json.dumps(report, indent=2) + '\n').encode())
    print(json.dumps({k:v for k,v in report.items()if k != 'resources'}, indent=2))


if __name__ == '__main__':
    main()
