"""Frozen full-suite, source, execution, resource and materiality gates."""
import ast
import hashlib
import json
import math
from pathlib import Path

from attention_skipscale_evidence import full_result, bound_result as retained_result
from verify_attention_precision import verify as verify_precision

ROOT = Path(__file__).resolve().parents[1]
PARENT = '9e922c3555484081ca45a99a90284e26eccab181fb677700d8227a4b3402412f'
TRIAL = 'f5d5b04bbd9eb44f6e06efdf3de1f5b83b3164a40d3526c42a406d99b58bbe66'
CUDA = '08fbf1d2380f957318ac501abbdbd9c9adff52bd64de63d936b3f727d6700464'
DRAFT = '7ef8ab72409ef250979fcd14802af38ec3b7e966f9c662d3ce535f650eced553'
PACKET = ROOT / 'results/attention-l1-wide-tile-preflight-20261029'
CHANGED = (3,5)
CONTROLS = tuple(i for i in range(12) if i not in CHANGED)


def read(path):
    return json.loads(path.read_bytes())


def source():
    raw = (ROOT / 'candidates/attention_active.py').read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    assert sha in (PARENT,TRIAL), ('Unreviewed source',sha)
    return raw,sha


def directory(sha):
    assert sha == TRIAL
    return ROOT / '.attention-wide-measurements' / sha


def bound_result(sha):
    if sha == PARENT:
        return retained_result(PARENT)
    folder = directory(sha)
    assert hashlib.sha256((folder / 'submission.py').read_bytes()).hexdigest() == sha
    launch,result = read(folder/'launch.json'),read(folder/'result.json')
    assert launch['source_sha256'] == sha
    assert result['id'] == launch.get('resolved_job_id',launch['job_id'])
    return result


def validate_packet():
    raw = (PACKET/'admission-v2/submission.py').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == TRIAL
    assert hashlib.sha256((PACKET/'admission-v2/kernel.cu').read_bytes()).hexdigest() == CUDA
    cuda = next(ast.literal_eval(n.value) for n in ast.parse(raw).body if isinstance(n,ast.Assign)
                and any(isinstance(t,ast.Name) and t.id=='CUDA_SRC' for t in n.targets))
    assert hashlib.sha256(cuda.encode()).hexdigest() == CUDA
    binding,admission,cpu,compiler,identity = [read(PACKET/n) for n in
        ['binding.json','admission-v2/binding.json','cpu-audit.json','compiler-review.json','evaluator-identity.json']]
    assert binding['draft_sha256'] == admission['draft_sha256'] == cpu['draft_sha256'] == DRAFT
    assert admission['source_sha256'] == TRIAL and admission['CUDA_sha256'] == CUDA
    assert admission['original_draft_restored_exactly'] and admission['Python_AST_unchanged']
    assert admission['host_macro_condition_parenthesized']
    assert cpu['ownership_and_bounds_cases'] == 450 and len(cpu['numerical_proxy_cases']) == 42
    assert compiler['disabled_draft_instruction_bodies_equal_retained']
    assert compiler['unchanged_L8_flat_and_L1_nonflat_instruction_bodies_equal']
    assert compiler['fixed_L1_flat_dynamic_shared_bytes'] == 84992
    assert all(arm['zero_stack_spill_fields'] for arm in compiler['arms'].values())
    assert identity['problem'] == 'muse_attention_decode'
    assert {r['file'] for r in identity['files']} == {'TASK.md','reference.py','checks.py','baseline.py','AGENTS.md'}
    assert all(r['exact'] and r['pinned_sha256']==r['fresh_sha256'] for r in identity['files'])
    for r in identity['files']:
        assert hashlib.sha256((ROOT/'muse_attention_decode'/r['file']).read_bytes()).hexdigest()==r['pinned_sha256']
    guard = read(PACKET/'guard-cpu-audit.json')
    assert guard['source_sha256'] == TRIAL and guard['positive_fixture_passed']
    assert len(guard['rejected_mutants']) >= 12
    verify_precision(raw)
    return raw


def verify_trial(result, verbose=True):
    body,benches = full_result(result)
    lines = [s.split() for s in body.get('log_tail','').splitlines() if s.startswith('ATTN_WIDE_RESOURCES ')]
    assert len(lines)==1, 'Missing or ambiguous resource marker'
    resources = [int(x) for x in lines[0][1:]]
    assert len(resources)==13
    assert 0 < resources[0] <= 128 and resources[1:3]==[0,0] and resources[3]>=2
    assert resources[4:12]==[84992,83968,128,2,128,128,200,128] and resources[12]>0
    profiles = {r['index']:r.get('report','').replace(' ','') for r in body['records'] if r.get('event')=='profile'}
    assert set(profiles)==set(range(12))
    assert all('attn_kernel_wide_l1<1,true' in profiles[i] for i in CHANGED), 'New L1-flat path absent'
    assert all(('attn_kernel_wide_l1<1,false' in profiles[i] or
                'attn_kernel_wide_l1<1,fals...' in profiles[i]) for i in (0,1,2,4)), 'Nonflat control path mismatch'
    assert all('attn_kernel_wide_l1<8,' in profiles[i] for i in range(6,12)), 'L8 control path mismatch'
    old,old_rows = full_result(bound_result(PARENT))
    previous = {r['index']:r['speedup'] for r in old_rows}
    ratios = {r['index']:r['speedup']/previous[r['index']] for r in benches}
    affected = math.prod(ratios[i] for i in CHANGED)**.5
    controls = math.prod(ratios[i] for i in CONTROLS)**(1/len(CONTROLS))
    summary = dict(source_sha256=TRIAL,affected_geomean=affected,controls_geomean=controls,
                   normalized_affected=affected/controls,aggregate_ratio=body['speedup']/old['speedup'],
                   per_case_ratios=ratios,resources=resources,statistical_significance_claim=False)
    if verbose:
        print(json.dumps(summary))
    assert affected>1.04 and affected/controls>1.04, 'Insufficient affected/control gain'
    assert all(ratios[i]/controls>=1.02 for i in CHANGED), 'Weak or regressing affected case'
    assert .95<=controls<=1.05, 'Controls drifted'
    assert body['speedup']>old['speedup']*1.015, 'Insufficient complete-track gain'
    return summary
