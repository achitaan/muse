"""Compile and archive one diagnosed L8 register-budget repair, without a GPU."""
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

from build_attention_two_score_pipeline import ROOT, OUT, freeze
from archive_attention_external_preflight import parse


def main():
    out = OUT.parent/'v2-register-repair'
    prefix = out/'compiler-prefix.cu'
    work = ROOT/'.tools/attention-two-score-v2-controls'
    controls = ['&attn_kernel_fp8pv<1,true,0>', '&attn_kernel_fp8pv<8,true,0>',
                '&attn_kernel_fp8pv<1,false,9>', '&attn_kernel_fp8pv<1,false,13>',
                '&attn_kernel_fp8pv<8,false,3>', '&attn_kernel_fp8pv<8,false,15>']
    if not work.with_suffix('.json').exists():
        command = [sys.executable, str(ROOT/'experiments/prepared/local_nvrtc_compile.py'),
                   '--source', str(prefix), '--minimal-std-tags', '--out', str(work),
                   '--define', 'INFINITY=__int_as_float(0x7f800000)']
        for name in controls+['&attn_kernel_two_score<8,true,0>']:
            command += ['--name', name]
        subprocess.run(command, cwd=ROOT, check=True)
    meta = json.loads(work.with_suffix('.json').read_bytes())
    assert meta['return_code'] == 0
    assert meta['original_source_sha256'] == hashlib.sha256(prefix.read_bytes()).hexdigest()
    log = work.with_suffix('.log').read_text()
    fields = re.findall(r'(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads', log)
    assert fields and all(tuple(map(int, row)) == (0,0,0) for row in fields)
    assert not re.search(r'warning.*C751', log)
    tool = ROOT/'.tools/cuda-compiler-13.0/nvidia/cu13/bin/nvdisasm.exe'
    sass = subprocess.check_output([str(tool), '--print-code', '--print-line-info', str(work.with_suffix('.cubin'))])
    entries = parse(sass.decode())
    old_sass = gzip.decompress((ROOT/'results/attention-external-finalize-preflight-20261029/v1/compiler/baseline.sass.gz').read_bytes()).decode()
    old = parse(old_sass)
    equality = {key: entries[key] == value for key,value in old.items()}
    assert len(equality) == 6 and all(equality.values()), equality
    sections = sass.decode().split('\t.section\t.text.')
    section = next(s for s in sections if 'attn_kernel_two_score' in s.split('\n',1)[0])
    code = re.findall(r'/\*[0-9a-f]+\*/\s+([^\n]+)', section)
    qk_bases = sorted(set(map(int, re.findall(r'QGMMA\.64x64x32\.F32\.E4M3\.E4M3 R(\d+),', section))))
    assert len(qk_bases) == 2 and abs(qk_bases[1]-qk_bases[0]) >= 32, qk_bases
    assert 'USETMAXREG.TRY_ALLOC.CTAPOOL UP0, 0xe8' in section
    assert 'gsb0, 0x1' in section and 'MUFU.EX2' in section
    regs = re.search(r'Compiling entry function[^\n]*attn_kernel_two_score[^\n]*\n.*?Used (\d+) registers', log, re.S)
    assert regs and int(regs[1]) == 168
    report = dict(CPU_only=True, GPU_jobs=0, exact_judge_toolchain=False,
        compiler_prefix_sha256=meta['original_source_sha256'],
        cubin_sha256=hashlib.sha256(work.with_suffix('.cubin').read_bytes()).hexdigest(),
        all_compiled_entries_zero_stack_and_spills=True, no_C751_warnings=True,
        offline_initial_registers=168, producer_target=32, consumer_target=232,
        initial_CTA_pool=168*384, redistributed_need=32*128+232*256,
        remaining_pool_registers=168*384-(32*128+232*256),
        score_accumulator_bases=qk_bases, distinct_32_register_score_ranges=True,
        partial_wait_and_exponential_instructions_present=True,
        six_control_instruction_and_branch_bodies_equal=equality,
        pipeline_instruction_count=len(code),
        limitations='Instruction presence is not a complete control-flow overlap proof. Actual H200 function attributes/occupancy and timing remain unverified; host dispatch not written yet.')
    for ext in ('.json','.log'):
        freeze(out/'compiler'/('controls'+ext), work.with_suffix(ext).read_bytes())
    freeze(out/'compiler/controls.sass.gz', gzip.compress(sass, mtime=0))
    for revision in ('v1','v2'):
        for ext in ('.json','.log'):
            freeze(OUT.parent/('v1' if revision=='v1' else 'v2-register-repair')/'compiler'/('initial'+ext),
                   (ROOT/('.tools/attention-two-score-'+revision+ext)).read_bytes())
    freeze(out/'compiler-review.json', (json.dumps(report,indent=2)+'\n').encode())
    print(json.dumps(report, indent=2))


if __name__ == '__main__': main()
