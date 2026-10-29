"""Summarize static proxy assembly, explicitly not runtime performance."""
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'STATE.json').exists())
PREFIX = ROOT / '.tools/nvrtc-preflight'


def summarize(stem, opcode_source):
    src = (PREFIX / (stem + '.cu')).read_text().splitlines()
    helper_line = next(i + 1 for i, line in enumerate(src) if opcode_source in line)
    raw = (PREFIX / (stem + '.sass')).read_bytes()
    counts, helper, examples = {}, {}, {}
    function, location = None, None
    for line in raw.decode().splitlines():
        section = re.search(r'^\s*\.section\s+\.text\.([^,]+)', line)
        if section:
            function = section.group(1) if 'moe_fused' in section.group(1) else None
            if function:
                counts.setdefault(function, Counter())
                helper.setdefault(function, Counter())
                examples.setdefault(function, [])
        loc = re.search(r'//## File "([^"]+)", line (\d+)', line)
        if loc:
            location = (loc.group(1), int(loc.group(2)))
        insn = re.search(r'^\s*/\*[0-9a-f]+\*/\s+(?:@!?U?P\d+\s+)?([A-Z][A-Z0-9_.]+)\b', line)
        if not insn or function is None:
            continue
        op = insn.group(1)
        counts[function][op] += 1
        if location and location[0].endswith('device-preflight.cu') and location[1] == helper_line:
            helper[function][op] += 1
            if op.startswith('STG') and len(examples[function]) < 4:
                examples[function].append(line.strip())
    assert len(counts) == 2 and all(helper.values())
    resources = subprocess.check_output([str(ROOT / '.tools/cuda-compiler-13.0/nvidia/cu13/bin/cuobjdump.exe'),
                                         '--dump-resource-usage', str(PREFIX / (stem + '.cubin'))])
    return dict(source_line_of_store_helper=helper_line,
                compiled_prefix_sha256=hashlib.sha256((PREFIX / (stem + '.cu')).read_bytes()).hexdigest(),
                cubin_sha256=hashlib.sha256((PREFIX / (stem + '.cubin')).read_bytes()).hexdigest(),
                sass_sha256=hashlib.sha256(raw).hexdigest(), sass_bytes=len(raw),
                per_kernel_static_opcodes={k: dict(v) for k, v in counts.items()},
                helper_static_opcodes={k: dict(v) for k, v in helper.items()},
                helper_store_examples=examples, resource_dump=resources.decode(errors='replace'))


def main():
    old = summarize('incumbent-main4', 'st.global.L2::cache_hint.b16')
    # The candidate's first b32 helper is the unchanged phase-B FP32 helper.
    # Select the uniquely named new helper by its declaration, not first opcode.
    lines = (PREFIX / 'epair-main.cu').read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if 'void epair_st_keep_u32(' in line)
    new_line = next(i + 1 for i in range(start, start + 8) if 'st.global.L2::cache_hint.b32' in lines[i])
    # Its full asm statement distinguishes the integer helper from the FP32 one.
    new = summarize('epair-main', lines[new_line - 1])
    report = dict(compiler='NVRTC13.0.88 Windows x86_64, sm_90a',
                  disassembler='NVIDIA nvdisasm/cuobjdump13.4.92',
                  cpu_compile_only=True, official_gpu_jobs=0, kernel_execution=False,
                  not_judge_NVCC_toolchain=True, baseline=old, candidate=new,
                  limitations=['Torch host launcher and resource query omitted',
                               'Explicit virtual type_traits/stdint headers and forward declaration used',
                               'Static instruction counts are not dynamic counts, memory traffic or timings',
                               'Stack/spill/shared fields are not substituted for official runtime attributes',
                               'Actual occupancy, graph correctness and performance remain unverified'])
    for entry in old['helper_static_opcodes'].values():
        assert entry.get('STG.E.U16', 0) > 0
    for entry in new['helper_static_opcodes'].values():
        assert entry.get('STG.E', 0) > 0 and entry.get('STG.E.U16', 0) == 0
    out = ROOT / 'results/moe-epair-nvrtc-preflight-20261029'
    out.mkdir(exist_ok=True)
    for stem in ('incumbent-main4', 'epair-main'):
        for suffix in ('.json', '.log'):
            data = (PREFIX / (stem + suffix)).read_bytes()
            target = out / (stem + suffix)
            assert not target.exists() or target.read_bytes() == data
            target.write_bytes(data)
        # Keep complete disassembly compressed, with deterministic gzip headers.
        data = gzip.compress((PREFIX / (stem + '.sass')).read_bytes(), mtime=0)
        target = out / (stem + '.sass.gz')
        assert not target.exists() or target.read_bytes() == data
        target.write_bytes(data)
    data = (json.dumps(report, indent=2) + '\n').encode()
    target = out / 'assembly-summary.json'
    assert not target.exists() or target.read_bytes() == data
    target.write_bytes(data)
    print(json.dumps(dict(baseline_helper=old['helper_static_opcodes'],
                          candidate_helper=new['helper_static_opcodes'], gpu_jobs=0)))


if __name__ == '__main__':
    main()
