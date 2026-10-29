"""Freeze CPU compiler and host-source evidence. Never loads or launches CUDA."""
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'STATE.json').exists())


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    work = ROOT / '.tools/moe-cluster-x'
    compile_base = ROOT / '.tools/nvrtc-preflight/moe-cluster-x-default-main'
    output = ROOT / 'results/moe-cluster-x-device-default-20261029'
    device = (work / 'device.cu').read_bytes()
    full = (work / 'full.cu').read_bytes()
    device_audit = json.loads((work / 'device-audit.json').read_bytes())
    host_audit = json.loads((work / 'host-audit.json').read_bytes())
    metadata = json.loads(compile_base.with_suffix('.json').read_bytes())
    assert sha(device) == device_audit['device_prefix_sha256'] == metadata['original_source_sha256']
    assert sha(full) == host_audit['full_CUDA_sha256']
    assert full.split(b'#include <torch/extension.h>', 1)[0] == device
    assert metadata['return_code'] == 0 and not metadata['gpu_kernels_loaded_or_launched']
    log = compile_base.with_suffix('.log').read_bytes()
    assert b'C751' not in log, 'WGMMA serialization disqualifies this preflight'
    assert log.count(b'16 bytes stack frame, 96 bytes spill stores, 160 bytes spill loads') == 2
    assert log.count(b'Used 96 registers, used 16 barriers') == 2
    cubin = compile_base.with_suffix('.cubin')
    tools = ROOT / '.tools/cuda-compiler-13.0/nvidia/cu13/bin'
    sass = subprocess.check_output([str(tools / 'nvdisasm.exe'), '--print-line-info', str(cubin)])
    resources = subprocess.check_output([str(tools / 'cuobjdump.exe'), '--dump-resource-usage', str(cubin)])
    assert resources.count(b'REG:96') == 2
    assert sass.count(b'EIATTR_CTA_PER_CLUSTER') == 2
    protected = json.loads((ROOT / '.tools/decoder-package-protected-sha256.json').read_bytes())
    assert all(sha((ROOT / name).read_bytes()) == value for name, value in protected.items())
    files = {name: (work / name).read_bytes() for name in (
        'device.cu', 'full.cu', 'edits.json', 'host-edits.json', 'device-audit.json', 'host-audit.json')}
    for source, target in [('build_moe_cluster_x_cuda.py', 'device-builder.py'),
                           ('build_moe_cluster_x_host.py', 'host-builder.py')]:
        files[target] = (ROOT / 'experiments/prepared' / source).read_bytes()
    files.update({'compile.json': compile_base.with_suffix('.json').read_bytes(), 'compile.log': log,
                  'complete.sass.gz': gzip.compress(sass, mtime=0), 'resources.txt': resources})
    binding = dict(parent_submission_sha256=device_audit['parent_submission_sha256'],
                   device_prefix_sha256=sha(device), full_CUDA_sha256=sha(full),
                   cubin_sha256=sha(cubin.read_bytes()), SASS_sha256=sha(sass),
                   files_sha256={name: sha(data) for name, data in files.items()},
                   protocol='Default counter release/acquire; remote DSM targets; full cluster lifetime barriers',
                   BIG_variants_compiled=2, reported_base_registers_each=96, offline_stack_bytes_each=16,
                   offline_spill_store_bytes_each=96, offline_spill_load_bytes_each=160,
                   WGMMA_serialization_warning=False, protected_files_exact=len(protected),
                   actual_runtime_attributes_verified=False, host_NVCC_compiled=False,
                   graph_capture_verified=False, numerical_correctness_verified=False,
                   performance_verified=False, challenge_submission_built=False,
                   kernel_execution=False, GPU_jobs=0)
    files['binding.json'] = (json.dumps(binding, indent=2) + '\n').encode()
    output.mkdir(exist_ok=True)
    for name, data in files.items():
        target = output / name
        assert not target.exists() or target.read_bytes() == data, ('Frozen evidence collision', name)
        target.write_bytes(data)
    print(json.dumps(dict(directory=str(output.relative_to(ROOT)), device=sha(device), full=sha(full),
                          protected_files_exact=len(protected), GPU_jobs=0)))


if __name__ == '__main__':
    main()
