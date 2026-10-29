"""Preserve direct-gather drafts and matched-control compiler evidence."""
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess

from build_moe_direct_stage_gather import ROOT,OUT,freeze,BASE,NATIVE
from archive_moe_private_fixup_preflight import resources


def main():
    packet=OUT.parent
    control=packet/'matched-control/kernel.cu'
    ledger=json.loads((packet/'v3-publisher-completion/binding.json').read_bytes())['edits']
    original=(OUT/'baseline.cu').read_text();matched=original;kept=[]
    for e in ledger:
        if e['name'] in ('namespace','namespace comment','host references','empty job guard','empty range guard'):
            assert matched.count(e['old'])==e['count'];matched=matched.replace(e['old'],e['new']);kept.append(e['name'])
    assert control.read_bytes()==matched.encode()
    freeze(control.parent/'binding.json',(json.dumps(dict(parent_source_sha256=BASE,parent_CUDA_sha256=NATIVE,
        CUDA_sha256=hashlib.sha256(control.read_bytes()).hexdigest(),shared_edits=kept,
        original_activation_gather_and_copy_preserved=True,CPU_only=True,GPU_jobs=0),indent=2)+'\n').encode())
    report=dict(parent_source_sha256=BASE,parent_CUDA_sha256=NATIVE,CPU_only=True,GPU_jobs=0,
        GPU_kernels_loaded=False,Python_submission_written=False,candidate_installed=False,revisions={})
    revisions=(('matched-control','control'),('v1','v1'),('v2-stripmined-copy','v2'),('v3-publisher-completion','v3'))
    tool=ROOT/'.tools/cuda-compiler-13.0/nvidia/cu13/bin/nvdisasm.exe'
    for version,stem in revisions:
        folder=packet/version
        work=ROOT/('.tools/moe-direct-stage-gather-'+stem+'/main')
        metadata=json.loads(work.with_suffix('.json').read_bytes())
        assert metadata['return_code']==0
        assert metadata['original_source_sha256']==hashlib.sha256((folder/'kernel.cu').read_bytes()).hexdigest()
        if version!='matched-control':
            binding=json.loads((folder/'binding.json').read_bytes());inverse=(folder/'kernel.cu').read_text()
            for e in reversed(binding['edits']):
                assert inverse.count(e['new'])==e['count'];inverse=inverse.replace(e['new'],e['old'])
            assert inverse==original
        for suffix in ('.cu','.json','.log'):
            freeze(folder/'compiler'/('main'+suffix),work.with_suffix(suffix).read_bytes())
        sass=subprocess.check_output([str(tool),'--print-code','--print-line-info',str(work.with_suffix('.cubin'))])
        freeze(folder/'compiler/main.sass.gz',gzip.compress(sass,mtime=0))
        log=work.with_suffix('.log').read_text()
        warnings=re.findall(r'[^\n]*C751[^\n]*',log)
        assert not warnings
        report['revisions'][version]=dict(resources=resources(log),serialization_warnings=warnings,
            cubin_sha256=hashlib.sha256(work.with_suffix('.cubin').read_bytes()).hexdigest(),
            exact_judge_toolchain=False)
    control_rows=report['revisions']['matched-control']['resources']
    for version,_ in revisions[1:]:
        rows=report['revisions'][version]['resources']
        assert all(r['registers']==96 for r in rows)
        assert [(r['stack_frame'],r['spill_store_bytes'],r['spill_load_bytes']) for r in rows]==[(48,316,668)]*2
    assert [(r['stack_frame'],r['spill_store_bytes'],r['spill_load_bytes']) for r in control_rows]==[(16,56,124)]*2
    # Independently compare strip-mined destinations/sources to the first CPU
    # model, including padding: no input entry is needed for an invalid row.
    vectors=0
    for count in range(1,65):
        padded=(count+7)//8*8
        for stage in range(16):
            seen=set()
            for lane in range(32):
                for r0 in range(0,padded,8):
                    row=r0+(lane&7)
                    for k in range(8):
                        byte_offset=r0*512+lane*16+k*512
                        idx=byte_offset//16
                        old_row=(idx//256)*8+(idx&7);old_k8=(idx>>3)&31
                        assert row==old_row and (lane>>3)*8+k*32==old_k8*8
                        assert idx not in seen;seen.add(idx);vectors+=1
            assert len(seen)==padded*32
    assert vectors==1179648
    report.update(stripmined_uint4_destinations_checked=vectors,
        decision='Park without a GPU check: static spill burden is higher than the matched control and recoverable complete-load time is unmeasured.',
        native_timing_or_actual_LocalSizeBytes_verified=False,
        known_cost='Direct stage copies issue8..64rounds instead of one bulk operation; v3 also waits for completion in the producer.',
        no_flag_or_register_sweep=True,source_and_precision_contract_preserved=True,
        unresolved='Offline stack/spill counters do not establish native H200 traffic or speed. No performance gain is claimed.')
    freeze(packet/'compiler-review.json',(json.dumps(report,indent=2)+'\n').encode())
    print(json.dumps({k:v for k,v in report.items() if k!='revisions'},indent=2))


if __name__=='__main__':main()
