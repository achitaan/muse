"""Source, native-resource and weighted-load evidence for the fixed Dense trial."""
import ast
import hashlib
import json
import math
from pathlib import Path
import re

from dense_cluster_evidence import PARENT,ROOT,full_result as prior_full_result,bound_result as parent_result
from verify_dense_cluster_evidence import verify_oracle

PRIOR='13224d30d3cf78a89bc731b3a6d036f4853c30cfb30529daa581050dd1f66c9b'
TRIAL='e48cba9a0a58be1953bf2f135ead88a5806ee079f5605f0b2dd348d1c8446652'
PRE_DIAGNOSTIC='7de2abc394419b2c80a866765dbd6b6a855828b8ab1aedbb0d35e7f862221139'
CUDA='69acf3cdf0b87c697c9c7c458ec1a56b92185ed4d9d695e544eb9e853d157541'
PRIOR_PACKET=ROOT/'results/dense-cluster-nearby-preflight-20261029'
DRAFT=ROOT/'results/dense-cluster-resource-draft-20261029'
PACKET=ROOT/'results/dense-cluster-resources-preflight-20261029'


def read(path):return json.loads(path.read_bytes())


def source():
    raw=(ROOT/'candidates/active.py').read_bytes();sha=hashlib.sha256(raw).hexdigest()
    assert sha in (PARENT,TRIAL),('Unreviewed Dense source',sha)
    return raw,sha


def directory(sha):
    assert sha in (PARENT,TRIAL)
    return ROOT/('.scored-measurements' if sha==PARENT else '.dense-cluster-resource-measurements')/sha


def bound_result(sha):
    if sha==PARENT:return parent_result(sha)
    folder=directory(sha)
    assert hashlib.sha256((folder/'submission.py').read_bytes()).hexdigest()==sha
    launch,result=read(folder/'launch.json'),read(folder/'result.json')
    assert launch['source_sha256']==sha and result['id']==launch.get('resolved_job_id',launch['job_id'])
    return result


def full_result(result):
    body,rows=prior_full_result(result)
    assert not body['checks'].get('flagged') and not body['checks'].get('rejected')
    ci=body['speedup_ci'];assert len(ci)==2 and all(math.isfinite(v) and v>0 for v in ci)
    assert ci[0]<=body['speedup']<=ci[1]
    profiles=[r for r in body['records'] if r.get('event')=='profile']
    assert sorted(r['index'] for r in profiles)==list(range(30))
    expected={r['index']:r['case']['weight'] for r in parent_result(PARENT)['result']['records'] if r.get('event')=='benchmark'}
    loads={}
    for r in rows:
        a,b,w=r['baseline_ns'],r['median_ns'],r['case']['weight']
        assert all(math.isfinite(x) and x>0 for x in (a,b,w)) and w==expected[r['index']]
        assert math.isclose(a/b,r['speedup'],rel_tol=1e-9)
        totals=loads.setdefault(r['case']['load'],[0.,0.]);totals[0]+=w*a;totals[1]+=w*b
    assert len(loads)==6
    assert math.isclose(math.prod(a/b for a,b in loads.values())**(1/6),body['speedup'],rel_tol=1e-9),'Wrong five-projection load aggregation'
    return body,rows,loads


def marker(log,name):
    lines=[line[len(name)+1:] for line in log.splitlines() if line.startswith(name+' ')]
    assert len(lines)==1,('Missing/duplicate marker',name)
    return lines[0]


def resources(log):
    values=json.loads(marker(log,'DENSE_CLUSTER_RESOURCES'))
    assert isinstance(values,list) and len(values)==2
    for kind,r in enumerate(values):
        cl,grid,smem=(3,108,230544) if kind==0 else (2,128,215184)
        assert isinstance(r,list) and len(r)==13 and all(type(v) is int for v in r)
        assert r[0]==kind and 0<r[1]<=128 and r[2:4]==[0,0]
        assert r[4]>=132 and r[5]>=grid//cl and r[6:10]==[smem,160,cl,grid]
        assert r[10]>=smem and r[11:]==[1,1]
    return values


def assignments(tree):
    return {t.id:n for n in tree.body if isinstance(n,ast.Assign) for t in n.targets if isinstance(t,ast.Name)}


def verify_source(raw):
    assert hashlib.sha256(raw).hexdigest()==TRIAL
    prior=(PRIOR_PACKET/'submission.py').read_bytes();assert hashlib.sha256(prior).hexdigest()==PRIOR
    baseline=(ROOT/'best/verified.py').read_bytes();assert hashlib.sha256(baseline).hexdigest()==PARENT
    assert raw.startswith(baseline) and prior.startswith(baseline)
    binding=read(DRAFT/'binding.json');assert binding['source_sha256']==PRE_DIAGNOSTIC and binding['parent_prepared_sha256']==PRIOR
    diagnostic=read(PACKET/'source-diagnostic-binding.json')
    assert diagnostic['source_sha256']==TRIAL and diagnostic['previous_draft_sha256']==PRE_DIAGNOSTIC
    assert diagnostic['diagnostic_compact_bytes_bound']<15000 and diagnostic['CUDA_unchanged']
    line,anchor,entry=(diagnostic[k] for k in ('removal_line','insertion_anchor','original_entry'))
    restored=raw.decode();assert restored.count(line+anchor)==restored.count(entry)==1
    restored=restored.replace(line+anchor,anchor).replace(entry,entry+line)
    assert restored.encode()==(DRAFT/'submission.py').read_bytes()
    edits=binding['edits'];assert len(edits)==4
    for e in reversed(edits):
        assert restored.count(e['new'])==1
        restored=restored.replace(e['new'],e['old'])
    assert restored.encode()==prior
    old,new=ast.parse(prior),ast.parse(raw);oa,na=assignments(old),assignments(new)
    old_cuda=ast.literal_eval(oa['_LG_CUDA'].value);cuda=ast.literal_eval(na['_LG_CUDA'].value)
    assert cuda==old_cuda+(DRAFT/'query.cu').read_text()
    assert hashlib.sha256(cuda.encode()).hexdigest()==CUDA
    old_cpp=ast.literal_eval(oa['_LG_CPP'].value);cpp=ast.literal_eval(na['_LG_CPP'].value)
    assert cpp==old_cpp+'\n#include <vector>\nstd::vector<int64_t> lg_info(int kind);\n'
    for name in ('_lg_key','_lg_run','_lg_input','_lg_verify','_lg_compare','kernel_fn'):
        a=[ast.dump(n) for n in old.body if isinstance(n,ast.FunctionDef) and n.name==name]
        b=[ast.dump(n) for n in new.body if isinstance(n,ast.FunctionDef) and n.name==name]
        assert a and a==b,name
    exports=[n for n in ast.walk(new) if isinstance(n,ast.keyword) and n.arg=='functions' and isinstance(n.value,ast.List)
             and all(isinstance(x,ast.Constant) for x in n.value.elts)]
    assert any(ast.literal_eval(n.value)==['gq','gqx','lg_info'] for n in exports)
    assert ast.literal_eval(na['_LG_CONFIGS'].value)==ast.literal_eval(oa['_LG_CONFIGS'].value)
    return dict(source_sha256=TRIAL,previous_prepared_sha256=PRIOR,previous_resource_draft_sha256=PRE_DIAGNOSTIC,original_parent_prefix_exact=True,
        inverse_restores_previous_source=True,all_timed_dispatch_numerical_paired_AST_unchanged=True,
        original_CUDA_exact_prefix=True,CUDA_sha256=CUDA,extra_host_export='lg_info(int kind)',CPU_only=True,GPU_jobs=0)


def validate_packet():
    raw=(PACKET/'submission.py').read_bytes();verify_source(raw)
    for name in ('cpu-audit.json','timing-unit-audit.json','nearby-selection-audit.json'):
        meta=read(PRIOR_PACKET/name);assert meta['source_sha256']==PRIOR
    cpu=read(PRIOR_PACKET/'cpu-audit.json')
    assert cpu['original_parent_prefix_exact'] and cpu['cuda_pieces_pinned'] and cpu['no_weight_requantization']
    assert cpu['old_wg_choices_proven_empty_from_published_log'] and cpu['output_ownership_cases']==256 and cpu['dispatch_cases']==5134
    assert len(cpu['numerical_cases'])==20
    nearby=read(PRIOR_PACKET/'nearby-selection-audit.json');units=read(PRIOR_PACKET/'timing-unit-audit.json')
    assert nearby['paired_cases']==20 and nearby['paired_repeats']==7 and len(nearby['guard_mutations_rejected'])==14
    assert units['cuda_event_multiplier']==1_000_000 and units['units']=='nanoseconds'
    compiler=read(DRAFT/'offline-device/compile-header-proxy.json')
    assert compiler['return_code']==0 and compiler['gpu_kernels_loaded_or_launched'] is False
    assert compiler['original_source_sha256']==hashlib.sha256((DRAFT/'offline-device/device-header-proxy.cu').read_bytes()).hexdigest()
    log=(DRAFT/'offline-device/compile-header-proxy.log').read_text()
    assert list(map(int,re.findall(r'Used (\d+) registers',log)))==[90,120]
    spills=re.findall(r'(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads',log)
    assert len(spills)==2 and all(tuple(map(int,r))==(0,0,0) for r in spills)
    identity=read(PACKET/'evaluator-identity.json');assert identity['problem']=='muse_bf16_proj'
    assert {r['file'] for r in identity['files']}=={'TASK.md','reference.py','checks.py','baseline.py','AGENTS.md'}
    for row in identity['files']:
        assert row['exact'] and row['pinned_sha256']==row['fresh_sha256']
        assert hashlib.sha256((ROOT/'muse_bf16_proj'/row['file']).read_bytes()).hexdigest()==row['pinned_sha256']
    guard=read(PACKET/'guard-CPU-admission-v2.json');assert guard['source_sha256']==TRIAL
    assert guard['positive_fixture_passed'] and len(guard['rejected_mutants'])>=25
    assert guard['guard_sha256']==hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    prior_guard=(ROOT/'scripts/verify_dense_cluster_evidence.py').read_text()
    node=next(n for n in ast.parse(prior_guard).body if isinstance(n,ast.FunctionDef) and n.name=='verify_oracle')
    assert guard['prior_oracle_source_sha256']==hashlib.sha256(ast.get_source_segment(prior_guard,node).encode()).hexdigest()
    return raw


def verify_trial(result,quiet=False):
    body,rows,loads=full_result(result);log=body.get('log_tail','')
    native=resources(log)
    choices=ast.literal_eval(marker(log,'DENSE_CLUSTER_CHOICES'))
    assert isinstance(choices,tuple) and choices and tuple(sorted(set(choices)))==choices
    assert set(choices)<={'qkv_gate','pooler'}
    evidence=json.loads(marker(log,'DENSE_CLUSTER_EVIDENCE'));assert set(choices)<=set(evidence)
    profiles={r['index']:r['report'].replace(' ','') for r in body['records'] if r.get('event')=='profile'}
    selected=[]
    for layer in choices:
        verify_oracle(layer,evidence[layer])
        index,symbol=(20,'gqx_kernel<128,1,8,3>') if layer=='qkv_gate' else (22,'gq_kernel<128,1,8,2>')
        assert symbol in profiles[index],('Selected native backend absent',layer)
        selected.append(index)
    for i,p in profiles.items():
        if i not in selected:assert 'gq_kernel<' not in p and 'gqx_kernel<' not in p,'Unexpected new backend on a control'
    old,old_rows,old_loads=full_result(bound_result(PARENT));old_map={r['index']:r for r in old_rows}
    ratios={r['index']:r['speedup']/old_map[r['index']]['speedup'] for r in rows}
    changed=math.prod(ratios[i] for i in selected)**(1/len(selected))
    controls=math.prod(v for i,v in ratios.items() if i not in selected)**(1/(30-len(selected)))
    gains={k:(a/b)/(old_loads[k][0]/old_loads[k][1]) for k,(a,b) in loads.items()}
    changed_loads={r['case']['load'] for r in rows if r['index'] in selected};assert len(changed_loads)==1
    weighted_controls=math.prod(v for k,v in gains.items() if k not in changed_loads)**(1/5)
    track=body['speedup']/old['speedup']
    summary=dict(choices=choices,native_resources=native,changed_projection_geomean=changed,
        control_projection_geomean=controls,normalized_projection_geomean=changed/controls,
        weighted_load_gains=gains,weighted_control_load_geomean=weighted_controls,
        aggregate_ratio=track,weighted_control_normalized_track_gain=track/weighted_controls,
        per_case_ratios=ratios,statistical_significance_claim=False)
    if not quiet:print(json.dumps(summary),flush=True)
    # Preserve every existing threshold; add weighted-control normalization.
    assert min(ratios[i] for i in selected)>=.98,'Selected projection regressed'
    assert changed>1.03 and changed/controls>1.03,'Insufficient affected/control improvement'
    assert track>1.015,'Insufficient aggregate gain'
    assert .95<=weighted_controls<=1.05,'Control load drift'
    assert track/weighted_controls>1.015,'Insufficient weighted-control track gain'
    return summary
