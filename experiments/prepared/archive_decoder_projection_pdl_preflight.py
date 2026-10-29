"""Archive the closed CPU scheduling screen, preserving both attempted forms."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'STATE.json').exists())
DEST = ROOT / 'results/decoder-projection-pdl-preflight-20261029'
DISASM = ROOT / '.tools/cuda-compiler-13.0/nvidia/cu13/bin/nvdisasm.exe'


def freeze(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    assert not path.exists() or path.read_bytes() == data, f'Frozen evidence collision: {path}'
    path.write_bytes(data)


def instructions(text):
    return [(int(pc, 16), op) for pc, op in re.findall(
        r'/\*([0-9a-f]+)\*/\s+(?:@!?P\d+\s+)?([A-Z][A-Z0-9_.]*)', text)]


def main():
    report = dict(GPU_jobs=0, GPU_kernels_loaded=False, candidate_installed=False,
                  exact_judge_toolchain=False, experiments=[])
    for version in ('v1', 'v2-ordered'):
        source = ROOT / ('.tools/decoder-pdl-preflight-' + version)
        destination = DEST / version
        freeze(destination / 'manifest.json', (source / 'manifest.json').read_bytes())
        for label in ('o1', 'o8', 'o16'):
            arms = {}
            for arm in ('baseline', 'wait_coherent'):
                stem = source / f'{label}-{arm}'
                cubin = stem.with_suffix('.cubin').read_bytes()
                process = subprocess.run([str(DISASM), '--print-code', '--print-line-info',
                                          str(stem.with_suffix('.cubin'))],
                                         capture_output=True, check=True)
                sass = process.stdout.decode()
                # Include device source, flags/adaptations, logs and full disassembly.
                # Cubins are reproducible local outputs; bind their exact bytes here.
                for suffix in ('.cu', '.json', '.log'):
                    freeze(destination / (stem.name + suffix), stem.with_suffix(suffix).read_bytes())
                freeze(destination / (stem.name + '.sass'), process.stdout)
                entries = []
                for section in sass.split('\t.section\t.text.'):
                    if '@"STO_CUDA_ENTRY STV_DEFAULT"' not in section:
                        continue
                    ops = instructions(section)
                    waits = [pc for pc, op in ops if op == 'ACQBULK']
                    loads = [pc for pc, op in ops if op.startswith('LDG.')]
                    assert loads
                    if arm == 'baseline':
                        assert not waits
                    else:
                        assert len(waits) == 1 and waits[0] < min(loads)
                    entries.append(dict(symbol=section.split('\n', 1)[0].split(',')[0],
                                        dependency_wait_PCs=waits,
                                        first_global_load_PC=min(loads),
                                        global_loads_before_wait=0 if waits else None,
                                        instruction_counts=dict(sorted(Counter(op for pc, op in ops).items()))))
                assert len(entries) == 2
                log = stem.with_suffix('.log').read_text()
                spills = re.findall(r'(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads', log)
                assert spills and all(tuple(map(int, row)) == (0, 0, 0) for row in spills)
                arms[arm] = dict(cubin_sha256=hashlib.sha256(cubin).hexdigest(),
                                 cubin_bytes=len(cubin),
                                 registers=list(map(int, re.findall(r'Used (\d+) registers', log))),
                                 stack_and_spill_fields_zero=True, kernels=entries)
            for baseline, changed in zip(arms['baseline']['kernels'], arms['wait_coherent']['kernels']):
                for op in set(baseline['instruction_counts']) | set(changed['instruction_counts']):
                    if op.startswith(('HMMA', 'STG', 'STS', 'LDS')):
                        assert baseline['instruction_counts'].get(op, 0) == changed['instruction_counts'].get(op, 0)
            report['experiments'].append(dict(version=version, tokens=label, arms=arms))
    report.update(decision='Park these direct wait-placement formulations without a GPU check.',
                  reason='The generated dependency wait precedes every global weight load in both formulations and all six selected/fallback kernels. Source prefetch placement did not survive device-code compilation.',
                  limitations=[
                      'This does not prove that every possible PDL implementation is ineffective.',
                      'No H200 launch, runtime occupancy, numerical correctness or performance was measured.',
                      'A future design needs a distinct compiler-verified independent-work mechanism and a credible complete-Decode opportunity.',
                  ])
    freeze(DEST / 'review.json', (json.dumps(report, indent=2) + '\n').encode())
    print(json.dumps(dict(experiments=len(report['experiments']), GPU_jobs=0,
                          dependency_wait_precedes_all_global_loads=True, decision=report['decision'])))


if __name__ == '__main__':
    main()
