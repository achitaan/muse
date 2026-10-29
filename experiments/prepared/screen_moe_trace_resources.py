"""Fixed CPU-only compiler screen; never load or launch a GPU kernel."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'STATE.json').exists())
PARENT = '331148716e8d1ba92b45cd29725be997d44b3a5235af74d75770bb9444100d50'
CUDA = '1e60cf469ebff8d0f1e63a223f41f041374a1f30f90e4201a819ee178353270c'
MACRO = '#define DBG(k) do { if (OKC_DEBUG && threadIdx.x == 0 && p.dbg) p.dbg[blockIdx.x * 8 + (k)] = gtimer(); } while (0)'
LOW = '#define DBG(k) do { if (OKC_DEBUG && threadIdx.x == 0 && p.dbg) { uint32_t t; asm volatile("mov.u32 %0, %globaltimer_lo;" : "=r"(t)); reinterpret_cast<uint32_t*>(p.dbg)[2 * (blockIdx.x * 8 + (k))] = t; } } while (0)'
REMAT = r'''#define DBG(k) do { if (OKC_DEBUG && threadIdx.x == 0) asm volatile("{ .reg .u64 ptr, stamp, off; .reg .u32 bid; .reg .pred active; ld.volatile.global.u64 ptr, [okc_trace_base]; setp.ne.u64 active, ptr, 0; @!active bra TRACE_DONE; mov.u32 bid, %ctaid.x; mul.wide.u32 off, bid, 64; add.u64 off, off, %0; add.u64 ptr, ptr, off; mov.u64 stamp, %globaltimer; st.global.u64 [ptr], stamp; TRACE_DONE: }" :: "n"((k) * 8) : "memory"); } while (0)'''
DECL = 'extern "C" { __device__ unsigned long long okc_trace_base = 0; }\n#define OKC_DEBUG 1'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def save(path, raw):
    assert not path.exists() or path.read_bytes() == raw, ('Frozen evidence collision', str(path))
    path.write_bytes(raw)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--compile', action='store_true', help='Run only the workspace NVRTC compiler on the CPU')
    parser.add_argument('--work-dir', default='.tools/nvrtc-preflight')
    parser.add_argument('--output-dir', default='results/moe-trace-resource-screen-20261029')
    args = parser.parse_args()
    work, out = (ROOT / args.work_dir).resolve(), (ROOT / args.output_dir).resolve()
    assert work.is_relative_to(ROOT) and out.is_relative_to(ROOT), 'Keep all artifacts in this workspace'
    if args.compile and out.exists() and any(out.iterdir()):
        parser.error('Choose an unused --output-dir and a fresh --work-dir for a new compilation record')
    work.mkdir(parents=True, exist_ok=True)
    raw = (ROOT / 'best/moe_verified.py').read_bytes()
    assert sha(raw) == PARENT
    tree = ast.parse(raw)
    source = next(n.value.value for n in tree.body if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == 'CUDA_SRC' for t in n.targets))
    assert sha(source.encode()) == CUDA
    assert source.count(MACRO) == source.count('#define OKC_DEBUG 0') == 1
    enabled = source.replace('#define OKC_DEBUG 0', '#define OKC_DEBUG 1', 1)
    variants = [
        ('moe-debug64-original', 'moe-debug64-main', enabled),
        ('moe-debug32-lowword', 'moe-debug32-main', enabled.replace(MACRO, LOW)),
        ('moe-debug-remat', 'moe-debug-remat-main', source.replace('#define OKC_DEBUG 0', DECL).replace(MACRO, REMAT)),
    ]
    assert variants[1][2].replace(LOW, MACRO) == enabled
    assert variants[2][2].replace(DECL, '#define OKC_DEBUG 0').replace(REMAT, MACRO) == source
    flags = ['OKC_NS=3', 'OKC_NSF=3', 'OKC_NW=16', 'OKC_FAST1=0', 'OKC_EPI4=0',
             'OKC_WGS=1', 'OKC_PRMTSC=1', 'OKC_DQ7=1']
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for source_stem, compiled_stem, candidate in variants:
        input_path, compiled_path = work / (source_stem + '.cu'), work / compiled_stem
        save(input_path, candidate.encode())
        if args.compile:
            command = [sys.executable, str(ROOT / 'experiments/prepared/local_nvrtc_compile.py'),
                       '--source', str(input_path), '--prefix-before-torch', '--minimal-std-tags',
                       '--forward-wgs-owner', '--name', '&okc::moe_fused<false>',
                       '--name', '&okc::moe_fused<true>', '--out', str(compiled_path)]
            for flag in flags:
                command.extend(['--define', flag])
            subprocess.run(command, check=True, cwd=ROOT)
        meta = json.loads(compiled_path.with_suffix('.json').read_bytes())
        assert meta['return_code'] == 0 and meta['original_source_sha256'] == sha(candidate.encode())
        assert meta['gpu_kernels_loaded_or_launched'] is False and meta['official_checks_used'] == 0
        dump = subprocess.check_output([
            str(ROOT / '.tools/cuda-compiler-13.0/nvidia/cu13/bin/cuobjdump.exe'),
            '--dump-resource-usage', str(compiled_path.with_suffix('.cubin'))]).decode()
        resource_rows = re.findall(r'REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)', dump)
        assert len(resource_rows) == 2
        rows.append(dict(variant=source_stem, original_cuda_sha256=sha(candidate.encode()),
                         cubin_sha256=sha(compiled_path.with_suffix('.cubin').read_bytes()),
                         compiled_prefix_sha256=meta['compiled_prefix_sha256'],
                         resource_dump=dump, per_BIG_resource_fields=[list(map(int, v)) for v in resource_rows]))
        for name, data in [(source_stem + '.cu', candidate.encode()),
                           (compiled_stem + '.json', compiled_path.with_suffix('.json').read_bytes()),
                           (compiled_stem + '.log', compiled_path.with_suffix('.log').read_bytes())]:
            save(out / name, data)
    baseline = json.loads((ROOT / 'results/moe-epair-nvrtc-preflight-20261029/assembly-summary.json').read_bytes())['baseline']
    report = dict(parent_submission_sha256=PARENT, parent_cuda_sha256=CUDA,
                  compiler='NVRTC13.0.88 Windows sm_90a; documented virtual headers/forward declaration',
                  baseline_compiled_prefix_sha256=baseline['compiled_prefix_sha256'],
                  baseline_resource_dump=baseline['resource_dump'], rows=rows,
                  official_GPU_jobs=0, kernel_execution=False, host_implementations_written=False,
                  latency_measurements=False, runtime_localSizeBytes_inferred=False,
                  conclusion='None removes the proxy stack increase: original16, DEBUG64/low32 each24, pointer-remat32. No GPU retry or source promotion.',
                  limitation='Offline STACK/LOCAL fields are not official runtime attributes; these builds do not identify the cause of historical GPU local8 bytes or prove all instrumentation designs impossible.')
    save(out / 'summary.json', (json.dumps(report, indent=2) + '\n').encode())
    print(json.dumps(dict(folder=str(out.relative_to(ROOT)),
                          per_BIG_resources=[(r['variant'], r['per_BIG_resource_fields']) for r in rows],
                          GPU_jobs=0)))


if __name__ == '__main__':
    main()
