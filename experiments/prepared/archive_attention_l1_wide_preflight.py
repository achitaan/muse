"""Bind the fixed draft to complete offline assembly/resources and CPU evidence."""
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess

from build_attention_l1_wide_tile import OUT, ROOT


def freeze(path, data):
    path.parent.mkdir(exist_ok=True, parents=True)
    assert not path.exists() or path.read_bytes() == data, f'Frozen evidence collision: {path}'
    path.write_bytes(data)


def kernels(sass):
    result = {}
    for section in sass.split('\t.section\t.text.'):
        head = section.split('\n', 1)[0]
        if 'attn_kernel_fp8pvI' not in head or 'STO_CUDA_ENTRY' not in section:
            continue
        key = head[head.index('attn_kernel_fp8pvI'):].split('EEv')[0]
        labels = list(dict.fromkeys(re.findall(r'^(\.L_[A-Za-z0-9_]+):', section, re.M)))
        label_map = {name: f'LABEL_{i}' for i, name in enumerate(labels)}
        code = re.findall(r'/\*[0-9a-f]+\*/\s+([^\n]+)', section)
        code = [re.sub(r'device_preflight_cu_[0-9a-f]+_\d+', 'device_preflight_cu_SYMBOL', x) for x in code]
        # Preserve distinct branch targets and their definition order while
        # ignoring the disassembler's global label numbering.
        code = [re.sub(r'\.L_[A-Za-z0-9_]+', lambda m: label_map[m[0]], x) for x in code]
        label_positions = [len(re.findall(r'/\*[0-9a-f]+\*/', section[:m.start()]))
                           for m in re.finditer(r'^\.L_[A-Za-z0-9_]+:', section, re.M)]
        result[key] = dict(instructions=code, label_instruction_positions=label_positions)
    assert len(result) == 3
    return result


def main():
    source = ROOT / '.tools/attention-l1-wide-preflight-v1'
    disassembler = ROOT / '.tools/cuda-compiler-13.0/nvidia/cu13/bin/nvdisasm.exe'
    report = dict(GPU_jobs=0, GPU_kernels_loaded=False, exact_judge_toolchain=False, arms={})
    parsed = {}
    for arm in ('baseline', 'wide', 'off'):
        for suffix in ('.cu', '.json', '.log'):
            freeze(OUT / 'compiler' / (arm + suffix), (source / (arm + suffix)).read_bytes())
        cubin = source / (arm + '.cubin')
        process = subprocess.run([str(disassembler), '--print-code', '--print-line-info', str(cubin)],
                                 capture_output=True, check=True)
        freeze(OUT / 'compiler' / (arm + '.sass.gz'), gzip.compress(process.stdout, mtime=0))
        parsed[arm] = kernels(process.stdout.decode())
        log = (source / (arm + '.log')).read_text()
        spills = re.findall(r'(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads', log)
        assert spills and all(tuple(map(int, x)) == (0,0,0) for x in spills)
        assert not re.search(r'warning.*C751[45]', log)
        report['arms'][arm] = dict(cubin_sha256=hashlib.sha256(cubin.read_bytes()).hexdigest(),
                                   registers_in_log=list(map(int, re.findall(r'Used (\d+) registers', log))),
                                   zero_stack_spill_fields=True,
                                   probed_dynamic_shared_bytes=sorted(set(map(int, re.findall(r'layout_bytes_probeILi(\d+)E', log)))),
                                   instruction_counts={k:len(v['instructions']) for k,v in parsed[arm].items()})
    assert parsed['baseline'] == parsed['off'], 'Disabled draft changed generated instruction bodies'
    unchanged = ('attn_kernel_fp8pvILi8ELb1ELi0E', 'attn_kernel_fp8pvILi1ELb0ELi1E')
    assert all(parsed['baseline'][k] == parsed['wide'][k] for k in unchanged)
    report.update(disabled_draft_instruction_bodies_equal_retained=True,
                  unchanged_L8_flat_and_L1_nonflat_instruction_bodies_equal=True,
                  normalization='Ignore disassembly local-label IDs and compiler-generated anonymous-namespace file hashes only',
                  fixed_L1_flat_dynamic_shared_bytes=84992,
                  retained_L1_flat_dynamic_shared_bytes=67584,
                  shared_memory_is_not_occupancy_validation=True,
                  candidate_installed=False,
                  remaining='Actual H200 resources, full numerical/cache/gate suite, real path execution and causal performance admission')
    assert 84992 in report['arms']['wide']['probed_dynamic_shared_bytes']
    assert 67584 in report['arms']['baseline']['probed_dynamic_shared_bytes']
    freeze(OUT / 'compiler-review.json', (json.dumps(report, indent=2) + '\n').encode())
    print(json.dumps({k:v for k,v in report.items() if k != 'arms'}))


if __name__ == '__main__':
    main()
