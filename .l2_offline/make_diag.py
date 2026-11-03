"""Reuse moediag7-style packaging, never L1's CUDA implementation."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
t=(ROOT/'.l1_offline/make_diag.py').read_text()
t=t.replace('moe_l1.py','moe_l2.py').replace('moediag_l1.py','moediag_l2.py').replace('L1','L2').replace('moediag_l1_','moediag_l2_')
t=t.replace('L2=640threads_112consumer_32producer_stagecapture',
            'L2=native80_fallback_GS1_576threads_weight3_x3or2_conserved_costs')
t=t.replace('640 threads, producer WG32/consumer WGs112, full packed-stage early release.',
            'Native65..80-token WGMMA jobs (N96 fallback);576 threads;3 weight slots,3x32KiB or2x48KiB x slots. No L1.')
# Preserve baseline-based NODUP/DUP3 routing transforms; add L2-specific metadata.
anchor="function('run_all',run)"
insert='''run=run.replace("jobs=jobs_from_counts(counts,'MAIN')", "jobs=jobs_from_counts(counts,'MAIN')\\n            native=[c for c in counts if 64<c<=80]")
run=run.replace("state=ns['_moe_sel_original_prepare'](raw);del raw", "ns['WGS_FIXW']=6144\\n    ns['WGS_FIXF']=2*ns['WGS_SLOTS']*4*6144+2*ns['WGS_SLOTS']*4+136*16\\n    state=ns['_moe_sel_original_prepare'](raw);del raw")
run=run.replace("f'job_MB={len(jobs)*WEIGHT_BYTES/1e6:.3f} '",
                "f'job_MB={len(jobs)*WEIGHT_BYTES/1e6:.3f} L2_jobs={len(jobs)-len(native)} wide={native} xslots={2 if native else 3} '")
run=run.replace('refs={};times={}', 'refs={};times={};narrow_refs={}')
run=run.replace("times[name,v,rep]=med", "times[name,v,rep]=med\\n                    narrow=_l2_narrow_rows(state,counts,ns['EMAX'])\\n                    if rep==1 and v=='BASE':narrow_refs[name]=narrow.clone()\\n                    if v=='L2':\\n                        refn=narrow_refs[name]\\n                        neq=torch.equal(narrow.contiguous().view(torch.uint8),refn.contiguous().view(torch.uint8))\\n                        ne=float((narrow-refn).norm()/refn.norm().clamp_min(1e-12))\\n                        log(f'MD_NARROW case={name} pass={rep} eq={int(neq)} rel_L2={ne:.3g}')")
helper="""def _l2_narrow_rows(state,counts,emax):
    # Untimed check: final output mixes wide/narrow routes, so inspect the
    # parent's per-route FP32 B outputs after Marlin/global/norm-weight scaling.
    y=state['scratch'][1][emax:emax+sum(counts)*4096].view(-1,4096)
    blocks=[];off=0
    for count in counts:
        if 0<count<=64:blocks.append(y[off:off+count])
        off+=count
    return torch.cat(blocks,dim=0) if blocks else y[:0]

"""
run=helper+run
'''+anchor
assert anchor in t;t=t.replace(anchor,insert)
exec(compile(t,str(ROOT/'.l2_offline/make_diag.py'),'exec'))
