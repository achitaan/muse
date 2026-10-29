"""Compile one fixed cluster4 residual/norm design without launching a GPU."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'STATE.json').exists())
BASE='d9fe7558624a1f070bbacb493b94302e90ff9bf77cdd8d4239f9d7a83ab39988'


def freeze(path,raw):
    path.parent.mkdir(parents=True,exist_ok=True)
    assert not path.exists() or path.read_bytes()==raw,str(path)
    path.write_bytes(raw)


def main():
    assert hashlib.sha256((ROOT/'best/decoder_verified.py').read_bytes()).hexdigest()==BASE
    source=ROOT/'experiments/prepared/decoder_cluster_norm.cu'
    raw=source.read_bytes();work=ROOT/'.tools/decoder-cluster-norm-v2/main'
    if not work.with_suffix('.json').exists():
        cmd=[sys.executable,str(ROOT/'experiments/prepared/local_nvrtc_compile.py'),
             '--source',str(source),'--out',str(work)]
        for kind in range(4):cmd+=['--name',f'&okc_cluster_norm::boundary<{kind}>']
        subprocess.run(cmd,cwd=ROOT,check=True)
    meta=json.loads(work.with_suffix('.json').read_bytes())
    assert meta['original_source_sha256']==hashlib.sha256(raw).hexdigest()
    assert meta['return_code']==0
    log=work.with_suffix('.log').read_text()
    fields=re.findall(r'(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads',log)
    assert fields and all(tuple(map(int,x))==(0,0,0) for x in fields)
    registers=list(map(int,re.findall(r'Used (\d+) registers',log)))
    out=ROOT/'results/decoder-cluster-norm-preflight-20261029/v2'
    for ext in ('.cu','.json','.log'):freeze(out/('compiler'+ext),work.with_suffix(ext).read_bytes())
    report=dict(parent_source_sha256=BASE,CUDA_sha256=hashlib.sha256(raw).hexdigest(),
        CPU_only=True,GPU_jobs=0,geometry=dict(cluster_CTAs=4,threads_per_CTA=256,columns_per_CTA=2048),
        full_row_norm_domain=8192,implicit_zero_tail_only_T_at_most_8=True,
        separate_stat_banks=True,final_DSM_lifetime_join=True,
        offline_register_counts=registers,all_zero_stack_and_spills=True,
        original_projections_and_dense_geometry_unchanged=True,
        candidate_installed=False,Python_submission_written=False,
        outstanding='Ownership/numerical model, source-bound whole-load opportunity screen, native cluster residency/graph probes, component pairing and full layer correctness/performance')
    freeze(out/'binding.json',(json.dumps(report,indent=2)+'\n').encode())
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
