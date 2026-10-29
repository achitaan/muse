"""Freeze the compile-only packet without promoting a runtime candidate."""
import hashlib
import json
from pathlib import Path
import re
import shutil

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'STATE.json').exists())
OUT=ROOT/'results/decoder-cooperative-boundary-screen-20261029'


def main():
    local=ROOT/'.tools/decoder-cooperative-boundary-preflight-v1'
    report=json.loads((local/'manifest.json').read_text())
    versions=[]
    for item in report['variants']:
        label=item['label']
        log=(local/f'{label}-cooperative.log').read_text()
        reg=list(map(int,re.findall(r'Used (\d+) registers',log)))
        fields=[list(map(int,m)) for m in re.findall(r'(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads',log)]
        assert len(reg)==len(fields)==2
        assert all(row==[0,0,0] for row in fields)
        for ext in ('.cu','.log','.json','.cubin'):
            src=local/f'{label}-cooperative{ext}';dst=OUT/'compiler'/src.name
            dst.parent.mkdir(parents=True,exist_ok=True)
            assert not dst.exists() or dst.read_bytes()==src.read_bytes()
            shutil.copyfile(src,dst)
        nt=item['threads'];grid=item['ctas']
        # Conservative register allocation granularity: eight regs/thread.
        rounded=max((r+7)//8*8 for r in reg)
        reg_blocks=65536//(rounded*nt)
        thread_blocks=2048//nt
        modeled=min(reg_blocks,thread_blocks,32,233472//item['dynamic_shared_bytes'])
        versions.append(dict(label=label,registers_packed_BF16=reg,stack_store_load_fields=fields,
            threads=nt,ctas=grid,dynamic_shared_bytes=item['dynamic_shared_bytes'],
            required_blocks_per_SM_for_132_SMs=(grid+131)//132,
            conservative_arithmetic_capacity_blocks_per_SM=modeled,
            runtime_occupancy_required=True,full_H200_residency_proven=False,
            source_sha256=item['source_sha256']))
    for src in (local/'manifest.json',):
        dst=OUT/'compiler'/src.name
        assert not dst.exists() or dst.read_bytes()==src.read_bytes()
        shutil.copyfile(src,dst)
    pinned=[]
    names=['fused_norm_residual.py','norm_residual.cuh','norm_residual.py']
    for name in names:
        part='jit/csrc' if name.endswith('.cuh') else 'ops'
        pinned.append(dict(path='pinned/'+name,sha256=hashlib.sha256((OUT/'pinned'/name).read_bytes()).hexdigest(),
            url=f'https://github.com/sgl-project/sglang/blob/a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f/python/sglang/kernels/{part}/muse_spark_v12/{name}'))
    result=dict(baseline_sha256=report['baseline_sha256'],CPU_only=True,GPU_jobs=0,
                active_candidate_changed=False,host_launcher_implemented=False,
                pinned_sources=pinned,variants=versions,
                inverse_edit_restores_selected_GEMV_device_prefix=True,
                decision='Continue bounded host/graph/precision admission work; not admitted for GPU yet')
    path=OUT/'screen.json';data=(json.dumps(result,indent=2)+'\n').encode()
    assert not path.exists() or path.read_bytes()==data
    path.write_bytes(data)
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
