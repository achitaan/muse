"""Bind fixed external-finalizer draft to full offline code/resource evidence."""
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess

from build_attention_external_finalize import ROOT,OUT,freeze


def parse(sass):
    entries={}
    for section in sass.split('\t.section\t.text.'):
        head=section.split('\n',1)[0]
        if 'STO_CUDA_ENTRY' not in section:continue
        match=re.search(r'attn_kernel_fp8pvILi\d+ELb[01]ELi\d+E',head)
        key=match[0] if match else ('attn_external_finalize8' if 'attn_external_finalize8' in head else None)
        if key is None:continue
        # nvdisasm appends private helper functions to the entry's text
        # section. A new entry renumbers their copies and changes namespace
        # encodings. Bind every defined target to its instruction offset,
        # including helpers: names may change, call/branch destinations may not.
        definitions=list(re.finditer(r'^([.$_A-Za-z][.$_A-Za-z0-9]*):',section,re.M))
        mapping={m[1]:'SYMBOL_AT_'+str(len(re.findall(r'/\*[0-9a-f]+\*/',section[:m.start()]))) for m in definitions}
        code=re.findall(r'/\*[0-9a-f]+\*/\s+([^\n]+)',section)
        code=[re.sub(r'[.$_A-Za-z][.$_A-Za-z0-9]*',lambda m:mapping.get(m[0],m[0]),x) for x in code]
        positions=sorted(set(int(v.removeprefix('SYMBOL_AT_')) for v in mapping.values()))
        entries[key]=dict(instructions=code,label_instruction_positions=positions)
    return entries


def main():
    work=ROOT/'.tools/attention-external-finalize-preflight-v1'
    tool=ROOT/'.tools/cuda-compiler-13.0/nvidia/cu13/bin/nvdisasm.exe'
    parsed={};report=dict(CPU_only=True,GPU_jobs=0,GPU_kernels_loaded=False,exact_judge_toolchain=False,arms={})
    for arm in ('baseline','external'):
        for extension in ('.cu','.log','.json'):
            freeze(OUT/'compiler'/(arm+extension),(work/(arm+extension)).read_bytes())
        cubin=work/(arm+'.cubin')
        process=subprocess.run([str(tool),'--print-code','--print-line-info',str(cubin)],capture_output=True,check=True)
        freeze(OUT/'compiler'/(arm+'.sass.gz'),gzip.compress(process.stdout,mtime=0))
        parsed[arm]=parse(process.stdout.decode())
        log=(work/(arm+'.log')).read_text()
        fields=re.findall(r'(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads',log)
        assert fields and all(tuple(map(int,x))==(0,0,0) for x in fields)
        assert not re.search(r'warning.*C751[45]',log)
        report['arms'][arm]=dict(cubin_sha256=hashlib.sha256(cubin.read_bytes()).hexdigest(),
            zero_stack_spill_fields=True,registers_in_log=list(map(int,re.findall(r'Used (\d+) registers',log))),
            dynamic_shared_probes=sorted(set(map(int,re.findall(r'layout_bytes_probeILi(\d+)E',log)))),
            instruction_counts={k:len(v['instructions']) for k,v in parsed[arm].items()})
    assert len(parsed['baseline'])==6 and len(parsed['external'])==8
    controls={k:parsed['baseline'][k]==parsed['external'][k] for k in parsed['baseline']}
    assert all(controls.values()),controls
    new=parsed['external']['attn_kernel_fp8pvILi8ELb1ELi16E']['instructions']
    old=parsed['baseline']['attn_kernel_fp8pvILi8ELb1ELi0E']['instructions']
    # FP32 payload REDG operations remain; only integer global counter arrivals
    # are expected to vanish from the deferred main. Shared pipeline barriers
    # and their waits are intentionally retained.
    old_integer_arrivals=[line for line in old if 'ATOMG' in line and 'F32' not in line]
    new_integer_arrivals=[line for line in new if 'ATOMG' in line and 'F32' not in line]
    assert old_integer_arrivals and not new_integer_arrivals
    report.update(six_instantiated_control_instruction_and_branch_bodies_equal=controls,
        new_main_integer_global_atomic_instructions=len(new_integer_arrivals),
        old_main_integer_global_atomic_instructions=len(old_integer_arrivals),
        FP32_payload_global_reds_retained=any('REDG' in line and 'F32' in line for line in new),
        normalization='Names of section-defined symbols only; preserve every target instruction offset and all appended helper instruction bodies. External symbols/constants are unchanged.',
        native_actual_occupancy_not_verified=True,candidate_installed=False,
        remaining='CPU row/index/replay and numerical model; exact native resource/dispatch gates; official full correctness/performance')
    assert report['FP32_payload_global_reds_retained']
    freeze(OUT/'compiler-review.json',(json.dumps(report,indent=2)+'\n').encode())
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
