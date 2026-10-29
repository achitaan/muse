"""Preserve the bounded compiler revisions and the unlaunched ownership study."""
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'STATE.json').exists())
OUT=ROOT/'results/moe-private-fixup-preflight-20261029'
DISASM=ROOT/'.tools/cuda-compiler-13.0/nvidia/cu13/bin/nvdisasm.exe'


def freeze(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    assert not path.exists() or path.read_bytes()==data,'Frozen compiler collision: '+str(path)
    path.write_bytes(data)


def resources(log):
    rows=[]
    for block in re.split(r'ptxas info\s+: Compiling entry function ',log)[1:]:
        name=block.split("'",2)[1]
        if 'moe_fused' not in name:continue
        fields=re.search(r'(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads',block)
        registers=re.search(r'Used (\d+) registers',block)
        assert fields and registers
        rows.append(dict(symbol=name,registers=int(registers[1]),
                         stack_frame=int(fields[1]),spill_store_bytes=int(fields[2]),spill_load_bytes=int(fields[3])))
    assert len(rows)==2
    return rows


def main():
    report=dict(GPU_jobs=0,GPU_kernels_loaded=False,candidate_installed=False,revisions={})
    for version in ('baseline','v1','v2','v3','v4'):
        stem=ROOT/('.tools/nvrtc-preflight/incumbent-main4' if version=='baseline'
                   else '.tools/moe-private-fixup-preflight-'+version+'/main')
        destination=OUT/'compiler'/version
        for suffix in ('.cu','.json','.log'):freeze(destination/('main'+suffix),stem.with_suffix(suffix).read_bytes())
        process=subprocess.run([str(DISASM),'--print-code','--print-line-info',str(stem.with_suffix('.cubin'))],
                               capture_output=True,check=True)
        freeze(destination/'main.sass.gz',gzip.compress(process.stdout,mtime=0))
        log=stem.with_suffix('.log').read_text()
        row=dict(resources=resources(log),cubin_sha256=hashlib.sha256(stem.with_suffix('.cubin').read_bytes()).hexdigest(),
                 serialization_warnings=re.findall(r'[^\n]*C751[^\n]*',log))
        assert not row['serialization_warnings']
        if version=='v4':
            row['layout_probe_values']=sorted(set(map(int,re.findall(r'private_layout_probeILi(\d+)E',log))))
            assert row['layout_probe_values']==[22248,231168]
            raw=(OUT/'v4/kernel.cu').read_bytes()
            proxy=(ROOT/'.tools/moe-private-fixup-preflight-v4/full-proxy.cu').read_bytes()
            addition=b'\ntemplate<int N> __global__ void private_layout_probe() {}\n'
            assert proxy.replace(addition,b'')==raw
            freeze(destination/'full-proxy.cu',proxy)
            row['raw_CUDA_sha256']=hashlib.sha256(raw).hexdigest()
            row['probe_only_adaptation']=addition.decode()
        report['revisions'][version]=row
    report.update(decision='Park without an H200 check: local-memory burden remains higher and exposed fixup benefit is unmeasured.',
                  current_CUDA_sha256=report['revisions']['v4']['raw_CUDA_sha256'],
                  extra_private_scratch_bytes=20971520,
                  shared_dynamic_probe_bytes=231168,
                  requires_separate_incumbent_and_candidate_scratch=True,
                  limitations=['Offline stack/spill fields are not actual H200 LocalSizeBytes, traffic or timing.',
                               'No numerical or replay correctness claim for the native draft.',
                               'Do not retune geometry, registers or commit-group flags merely to force admission.'])
    freeze(OUT/'compiler-review.json',(json.dumps(report,indent=2)+'\n').encode())
    print(json.dumps({k:v for k,v in report.items() if k!='revisions'}))


if __name__=='__main__':main()
