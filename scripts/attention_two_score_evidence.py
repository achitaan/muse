"""Frozen L8 pipeline provenance, complete-suite and causal gain gates."""
import hashlib
import json
import math
from pathlib import Path
import sys

from attention_skipscale_evidence import full_result, bound_result as retained_result

ROOT=Path(__file__).resolve().parents[1]
PARENT='9e922c3555484081ca45a99a90284e26eccab181fb677700d8227a4b3402412f'
TRIAL='c51f843e34f0780ed77483bcb864574e2dc3d69f0b0deb987eb2b75ffe1f3d61'
CUDA='375726fddea48bc8641167f3c9b4a458b585125ddb4d3987b4e766e88a036816'
PACKET=ROOT/'results/attention-two-score-pipeline-preflight-20261029'
CHANGED=(9,11)
CONTROLS=tuple(i for i in range(12) if i not in CHANGED)


def read(path):return json.loads(path.read_bytes())


def source():
    raw=(ROOT/'candidates/attention_active.py').read_bytes()
    sha=hashlib.sha256(raw).hexdigest()
    assert sha in (PARENT,TRIAL),('Unreviewed source',sha)
    return raw,sha


def directory(sha):
    return ROOT/'results/attention-segment-repair-20261028/publication'/sha if sha==PARENT else ROOT/'.attention-two-score-measurements'/sha


def bound_result(sha):
    if sha==PARENT:return retained_result(sha)
    folder=directory(sha)
    assert hashlib.sha256((folder/'submission.py').read_bytes()).hexdigest()==sha
    launch=read(folder/'launch.json');result=read(folder/'result.json')
    assert launch['source_sha256']==sha and result['id']==launch.get('resolved_job_id',launch['job_id'])
    return result


def validate_packet():
    raw=(PACKET/'admission-v3/submission.py').read_bytes()
    assert hashlib.sha256(raw).hexdigest()==TRIAL
    assert hashlib.sha256((PACKET/'admission-v3/kernel.cu').read_bytes()).hexdigest()==CUDA
    sys.path.insert(0,str(ROOT/'experiments/prepared'))
    from audit_attention_two_score_CPU import source_proof
    proof=source_proof()
    assert proof['source_sha256']==TRIAL and proof['CUDA_sha256']==CUDA
    cpu=read(PACKET/'source-coverage-CPU-proof.json')
    assert cpu['source_sha256']==TRIAL and cpu['partition_cases']==360 and cpu['changing_grid_epochs']==256
    event=read(ROOT/'results/attention-two-score-pipeline-model-20261029/model.json')
    assert event['event_cases']==445 and len(event['rejected_hazards'])==7 and event['max_committed_groups']==2
    compiled=read(PACKET/'v2-register-repair/compiler-review.json')
    assert compiled['all_compiled_entries_zero_stack_and_spills'] and compiled['no_C751_warnings']
    assert compiled['offline_initial_registers']==168 and compiled['redistributed_need']==63488
    assert compiled['initial_CTA_pool']==64512 and compiled['distinct_32_register_score_ranges']
    assert len(compiled['six_control_instruction_and_branch_bodies_equal'])==6
    assert all(compiled['six_control_instruction_and_branch_bodies_equal'].values())
    identity=read(PACKET/'evaluator-identity.json')
    assert identity['problem']=='muse_attention_decode'
    assert {row['file'] for row in identity['files']}=={'TASK.md','reference.py','checks.py','baseline.py','AGENTS.md'}
    for row in identity['files']:
        assert row['exact'] and row['fresh_sha256']==row['pinned_sha256']
        assert hashlib.sha256((ROOT/'muse_attention_decode'/row['file']).read_bytes()).hexdigest()==row['pinned_sha256']
    guard=read(PACKET/'guard-CPU-audit.json')
    assert guard['source_sha256']==TRIAL and guard['positive_fixture_passed'] and len(guard['rejected_mutants'])>=20
    assert guard['guard_sha256']==hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return raw


def verify_trial(result,verbose=True):
    body,rows=full_result(result)
    markers=[line.split() for line in body.get('log_tail','').splitlines() if line.startswith('ATTN_TWO_SCORE_RESOURCES ')]
    assert len(markers)==1,'Missing/ambiguous native admission marker'
    r=list(map(int,markers[0][1:]))
    assert len(r)==14
    assert 0<r[0]<=80 and r[1:3]==[0,0] and r[3]>=2
    assert r[4:9]==[168,0,0,1,114688]
    assert r[9]>=132 and r[10:]==[384,4,32,232]
    profiles={row['index']:row.get('report','').replace(' ','') for row in body['records'] if row.get('event')=='profile'}
    assert set(profiles)==set(range(12))
    # This exact source instantiates only <8,true,0> of the new kernel. Its
    # unique name remains evidence even when official report columns truncate
    # template arguments. Native resource and exact-source proofs are mandatory.
    for i in CHANGED:
        assert 'attn_kernel_two_score' in profiles[i],'New pipeline did not execute'
        assert 'attn_kernel_fp8pv<' not in profiles[i],'Duplicate old/new main execution'
        assert 'attn_external_finalize8' not in profiles[i],'Rejected finalizer mechanism unexpectedly executed'
    for i in CONTROLS:
        assert 'attn_kernel_two_score' not in profiles[i],'Control changed'
        assert 'attn_kernel_fp8pv<' in profiles[i],'Control backend missing'
    old,old_rows=full_result(bound_result(PARENT))
    prior={row['index']:row['speedup'] for row in old_rows}
    ratios={row['index']:row['speedup']/prior[row['index']] for row in rows}
    affected=math.prod(ratios[i] for i in CHANGED)**.5
    controls=math.prod(ratios[i] for i in CONTROLS)**(1/len(CONTROLS))
    summary=dict(source_sha256=TRIAL,affected_case_geomean=affected,controls_case_geomean=controls,
        normalized_affected=affected/controls,aggregate_ratio=body['speedup']/old['speedup'],
        per_case_ratios=ratios,resources=r,statistical_significance_claim=False)
    if verbose:print(json.dumps(summary),flush=True)
    assert .95<=controls<=1.05,'Controls drifted; retain baseline'
    assert affected>=1.02 and affected/controls>=1.02,'Insufficient affected/control gain'
    assert all(ratios[i]/controls>=1.01 for i in CHANGED),'Weak/regressing affected case'
    assert summary['aggregate_ratio']>=1.005,'Insufficient weighted track gain'
    return summary
