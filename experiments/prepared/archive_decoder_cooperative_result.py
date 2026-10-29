"""Preserve terminal cooperative trial evidence without promoting its score."""
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import base64
import zlib

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'STATE.json').exists())
sys.path.insert(0,str(ROOT/'scripts'))
from decoder_cooperative_evidence import TRIAL,PARENT,directory,bound_result,full_result,markers


def freeze(path,raw):
    path.parent.mkdir(parents=True,exist_ok=True)
    assert not path.exists() or path.read_bytes()==raw,'Frozen evidence collision: '+str(path)
    path.write_bytes(raw)


def main():
    result=bound_result(TRIAL);body,rows,loads=full_result(result)
    assert result['id']=='7f85bd70a3484dc7' and result['mode']=='check'
    assert hashlib.sha256((ROOT/'candidates/decode_active.py').read_bytes()).hexdigest()==PARENT
    protected=json.loads((ROOT/'.tools/decoder-package-protected-sha256.json').read_bytes())
    assert all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h for n,h in protected.items())
    log=body['log_tail'];assert markers(log,'CB_SELECTED')==[[]]
    pairs=markers(log,'CB_PAIR');assert len(pairs)==2 and all(p['keep'] is False and p['wins']==0 for p in pairs)
    comparison=[]
    for p in pairs:
        ratios=[s['old']/s['new'] for s in p['samples']]
        gain=math.prod(ratios)**(1/16)
        assert math.isclose(gain,p['gain'],rel_tol=1e-12) and gain<1
        comparison.append(dict(family=p['family'],fmt=p['fmt'],old_over_new_geomean=gain,
            new_over_old_geomean=1/gain,new_slowdown_percent=(1/gain-1)*100,
            old_median_ns=statistics.median(s['old'] for s in p['samples']),
            new_median_ns=statistics.median(s['new'] for s in p['samples']),
            winning_pairs=p['wins'],sample_pairs=len(p['samples']),cache=p['cache'],
            independent_GPU_jobs=1,statistical_significance_claim=False))
    resources=markers(log,'CB_RESOURCES')
    assert len(resources)==4 and {(r['family'],r['fmt']) for r in resources}=={(1,0),(1,1),(8,0),(8,1)}
    for r in resources:
        q=r['query'];family=r['family']
        assert q['1']==q['7']==1 and q['3']==q['4']==0 and q['6']==132
        assert (128 if family==1 else 512)<=q['5']*q['6']
        assert q['8']==(16384 if family==1 else 4096)
    probes=markers(log,'CB_PROBES_Z');decoded=[];maximum=[0.,0.,0.]
    assert len(probes)==4
    for p in probes:
        cases=json.loads(zlib.decompress(base64.b64decode(p['payload'],validate=True)))
        assert len(cases)==p['cases']==(8 if p['family']==1 else 32)
        expected={(m,f) for m in ((1,) if p['family']==1 else range(2,9)) for f in ('random','zero','outlier','tiny')}
        expected|={(p['family'],f'fresh_graph_replay_{i}') for i in range(4)}
        assert {(c['m'],c['fixture']) for c in cases}==expected
        for c in cases:
            assert c['errors'][3]==[0.,0.]
            for i,values in enumerate(c['errors'][:3]):
                assert all(math.isfinite(v) and 0<=v<=(1e-3 if i==0 else 2e-5) for v in values)
                maximum[i]=max(maximum[i],*values)
        decoded.append(dict(family=p['family'],fmt=p['fmt'],cases=cases))
    assert sum(len(p['cases']) for p in decoded)==80
    profiles=[r for r in body['records'] if r.get('event')=='profile']
    assert len(profiles)==12 and all('cb_smallm2_gemv' not in r['report'] and 'cb_v18s_gemv' not in r['report'] for r in profiles)
    setup=next(r for r in body['records'] if r['event']=='setup')
    source=directory(TRIAL);out=ROOT/'results/decoder-cooperative-boundary-check-20261029'
    for path in source.rglob('*'):
        if path.is_file(): freeze(out/path.relative_to(source),path.read_bytes())
    archive=ROOT/'autoresearch-results/archive/20261029-125933'
    for path in archive.rglob('*'):
        if path.is_file(): freeze(out/'controller'/path.relative_to(archive),path.read_bytes())
    freeze(out/'decoded-synthetic-probes.json',(json.dumps(decoded,indent=2)+'\n').encode())
    freeze(out/'restored-source-proof.json',(json.dumps(dict(all_protected_hashes_match=True,count=len(protected),hashes=protected),indent=2)+'\n').encode())
    quota=json.loads((source/'quota-before-launch.json').read_bytes())
    expected={p['name']:dict(checks=p['limits']['checks']['left']-(1 if p['name']=='muse_decode_layer' else 0),
                              scores=p['limits']['scored']['left']) for p in quota['response']['problems']}
    report=dict(job_id=result['id'],source_sha256=TRIAL,CUDA_sha256='51180052ffd10dfb11ab1bb747c11192922c7effdd2669b4389185184028e4b1',
        full_tests=13,full_benchmarks=12,integrity_ok=body['checks']['ok'],GPU=body['gpu'],
        full_suite_scope='Retained timed GPU paths: both experimental families disabled by fixed paired component gate',
        speedup=body['speedup'],speedup_ci=body['speedup_ci'],new_kernel_speedup_claim=False,
        native_resources=resources,synthetic_correctness_probe_cases=80,changing_input_graph_replays=16,
        max_global_or_worst_row_relL2_output_residual_router=maximum,
        BF16_projection_exact_for_all_synthetic_probes=True,component_comparisons=comparison,
        selected_families=[],experimental_kernel_profile_cases=0,
        prepare_seconds=setup['prepare_s'],preparation_margin_under_300_seconds=300-setup['prepare_s'],
        isolated_packaging_preparation_savings_claim=False,
        controller_run_id='ee1ef27a72114dc2b306b25e192977db',trial_commit='c4f25090f262b583cc77162d77ad3420b321d02f',
        revert_commit='bd2a43ddea814091cfa96ab612abc1ebdfed8805',controller_archive='20261029-125933',
        retained_source_sha256=PARENT,all_44_protected_hashes_restored=True,new_scored_submissions=0,
        derived_quotas_after_accepted_check=expected,quota_scope='Computed from fresh pre-launch API minus this accepted check; not a fresh post-result API snapshot',
        next_new_check_not_before_utc='2026-10-29T13:50:40.319601+00:00',
        decision='Close cooperative O1/O8 whole-grid boundary fusion; retain D9; no score, noise rerun, tile sweep or transfer',
        limitations='Warm synthetic component measurements are not cold-L2 whole-layer gains. Full-suite point changes use retained paths. Barrier cost has not been separately measured.')
    freeze(out/'review.json',(json.dumps(report,indent=2)+'\n').encode())
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
