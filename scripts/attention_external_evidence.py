"""Source-bound L8 external-finalization coverage, execution and gain gates."""
import ast
import contextlib
import copy
import hashlib
import io
import json
import math
from pathlib import Path
import re

from attention_skipscale_evidence import full_result, bound_result as retained_result
from verify_attention_precision import verify as verify_precision

ROOT=Path(__file__).resolve().parents[1]
PARENT='9e922c3555484081ca45a99a90284e26eccab181fb677700d8227a4b3402412f'
DRAFT='4cbe02668aaf52bd5c514f470c1cec3b7b433932fbbed5a87abc7078daa74432'
TRIAL='88026272155440d2dce8c565ac2bcbbe28a4fef97c0cf4909a3b5d6a90fdf794'
CUDA='c02c053a7afcab82fb1d09df7b74361855d6c24c6b664633db3dec675d06eeb7'
PACKET=ROOT/'results/attention-external-finalize-preflight-20261029'
CHANGED=(9,11)
CONTROLS=tuple(i for i in range(12) if i not in CHANGED)


def read(path):return json.loads(path.read_bytes())


def source():
    raw=(ROOT/'candidates/attention_active.py').read_bytes();sha=hashlib.sha256(raw).hexdigest()
    assert sha in (PARENT,TRIAL),('Unreviewed source',sha)
    return raw,sha


def directory(sha):
    assert sha==TRIAL
    return ROOT/'.attention-external-measurements'/sha


def bound_result(sha):
    if sha==PARENT:return retained_result(PARENT)
    folder=directory(sha)
    assert hashlib.sha256((folder/'submission.py').read_bytes()).hexdigest()==sha
    launch,result=read(folder/'launch.json'),read(folder/'result.json')
    assert launch['source_sha256']==sha
    assert result['id']==launch.get('resolved_job_id',launch['job_id'])
    return result


def assignment(tree,name):
    nodes=[n for n in tree.body if isinstance(n,ast.Assign)
           and any(isinstance(t,ast.Name) and t.id==name for t in n.targets)]
    assert len(nodes)==1
    return nodes[0]


def verify_source(raw):
    assert hashlib.sha256(raw).hexdigest()==TRIAL
    parent=(ROOT/'best/attention_checked.py').read_bytes()
    assert hashlib.sha256(parent).hexdigest()==PARENT
    parent_tree=ast.parse(parent);tree=ast.parse(raw)
    cuda=ast.literal_eval(assignment(tree,'CUDA_SRC').value)
    assert hashlib.sha256(cuda.encode()).hexdigest()==CUDA
    draft=(PACKET/'v1/submission.py').read_bytes()
    assert hashlib.sha256(draft).hexdigest()==DRAFT
    d_tree=ast.parse(draft);d_cuda=ast.literal_eval(assignment(d_tree,'CUDA_SRC').value)
    old_cuda=ast.literal_eval(assignment(parent_tree,'CUDA_SRC').value)
    assert cuda.startswith(d_cuda)
    # Independently undo every exact patch and compare with the authoritative
    # parent literal. Do not accept boolean claims in a builder manifest.
    recovered=d_cuda
    for edit in reversed(read(PACKET/'v1/binding.json')['edits']):
        assert recovered.count(edit['new'])==edit['count']
        recovered=recovered.replace(edit['new'],edit['old'])
    assert recovered==old_cuda
    # Every instruction-producing source statement before completion,
    # including global FP32 partial reductions and producer drain, is exact.
    start='template <int L, bool FLAT, int ROLE, int MODE = 0>\n'
    stop='  // (last-CTA finalizes): no per-thread fences;'
    deferred='  if constexpr (!(FLAT && L == 8 && (MODE & 16))) {\n'
    numerical=old_cuda[old_cuda.index(start):old_cuda.index(stop)]
    assert numerical==d_cuda[d_cuda.index(start):d_cuda.index(deferred)]
    assert set(map(int,re.findall(r'MODE\s*&\s*(\d+)',numerical)))=={3,4,8}
    assert all((16&mask)==(0&mask) for mask in (3,4,8))
    barrier="     // consumers' shared memory is reused by the next segment\n   __syncthreads();"
    assert old_cuda.count(barrier)==d_cuda.count(barrier)==1
    producer="        __syncthreads();   // matches the consumers' end-of-segment barrier"
    assert old_cuda.count(producer)==d_cuda.count(producer)==1
    def finalizer(text,name):
        anchor='template <int L, int NT>\n__device__ __forceinline__ void '+name+'('
        a=text.index(anchor);b=text.index('\n}\n',a)+3
        return text[a:b]
    original_fin=finalizer(old_cuda,'fin_nc1')
    new_fin=finalizer(d_cuda,'external_fin_nc1')
    sync='    __syncwarp(0xffffffffu); // all eight lanes read lsum before part0 clears it\n'
    assert new_fin.count(sync)==1
    assert new_fin.replace('external_fin_nc1(','fin_nc1(',1).replace(sync,'',1)==original_fin
    # Restore only the bounded native literals/build admission to check every
    # other Python node, including the large-L8 fallback and index domains.
    restored=copy.deepcopy(tree)
    for i,node in enumerate(restored.body):
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ('CUDA_SRC','CPP_SRC') for t in node.targets):
            name=node.targets[0].id;restored.body[i]=copy.deepcopy(assignment(parent_tree,name))
        elif isinstance(node,ast.FunctionDef) and node.name=='_build':
            restored.body[i]=copy.deepcopy(next(n for n in parent_tree.body if isinstance(n,ast.FunctionDef) and n.name=='_build'))
    assert ast.dump(restored)==ast.dump(parent_tree)
    with contextlib.redirect_stdout(io.StringIO()):verify_precision(parent)
    return dict(exact_mainloop_before_completion=True,exact_new_finalizer_math_except_warp_ordering=True,
                full_segment_barriers_retained=True,all_other_Python_nodes_and_fallbacks_exact=True,
                original_BF16_norm_RoPE_attention_and_final_output_boundaries_preserved=True)


def validate_packet():
    raw=(PACKET/'admission-v2/submission.py').read_bytes();verify_source(raw)
    assert hashlib.sha256((PACKET/'admission-v2/kernel.cu').read_bytes()).hexdigest()==CUDA
    draft,admission,model,compiler,identity=[read(PACKET/n) for n in
        ('v1/binding.json','admission-v2/binding.json','v1/CPU-model.json','v1/compiler-review.json','evaluator-identity.json')]
    assert draft['source_sha256']==admission['draft_sha256']==model['source_sha256']==DRAFT
    assert admission['source_sha256']==TRIAL and admission['CUDA_sha256']==CUDA
    assert model['ownership_shapes']==44 and model['sequential_old_external_epoch_cases']==35
    assert len(model['numerical_proxy_cases'])==45 and all(model['mutants_detected'].values())
    assert model['pair_sensitive_index_mutant_detected']
    assert len(compiler['six_instantiated_control_instruction_and_branch_bodies_equal'])==6
    assert all(compiler['six_instantiated_control_instruction_and_branch_bodies_equal'].values())
    assert compiler['new_main_integer_global_atomic_instructions']==0 and compiler['FP32_payload_global_reds_retained']
    assert all(a['zero_stack_spill_fields'] for a in compiler['arms'].values())
    assert identity['problem']=='muse_attention_decode'
    assert {r['file'] for r in identity['files']}=={'TASK.md','reference.py','checks.py','baseline.py','AGENTS.md'}
    for row in identity['files']:
        assert row['exact'] and row['fresh_sha256']==row['pinned_sha256']
        assert hashlib.sha256((ROOT/'muse_attention_decode'/row['file']).read_bytes()).hexdigest()==row['pinned_sha256']
    guard=read(PACKET/'guard-CPU-audit.json')
    assert guard['source_sha256']==TRIAL and guard['positive_fixture_passed'] and len(guard['rejected_mutants'])>=15
    return raw


def verify_trial(result,verbose=True):
    body,rows=full_result(result)
    lines=[line.split() for line in body.get('log_tail','').splitlines() if line.startswith('ATTN_EXTERNAL_RESOURCES ')]
    assert len(lines)==1,'Missing/ambiguous native resource marker'
    r=[int(v) for v in lines[0][1:]]
    assert len(r)==18
    assert 0<r[0]<=80 and r[1:3]==[0,0] and r[3]>=2
    assert 0<r[4]<=80 and r[5:7]==[0,0] and r[7]>=2
    assert r[8]==114688 and 0<r[9]<=64 and r[10:12]==[0,0]
    assert r[12]>=1 and r[13]>=132 and r[14:]==[64,384,128,16]
    profiles={row['index']:row.get('report','').replace(' ','') for row in body['records'] if row.get('event')=='profile'}
    assert set(profiles)==set(range(12))
    for i in CHANGED:
        assert 'attn_kernel_fp8pv<8,true,16>' in profiles[i],'Deferred main path absent'
        assert 'attn_external_finalize8' in profiles[i],'Parallel finalizer absent'
        assert 'attn_kernel_fp8pv<8,true,0>' not in profiles[i],'Retained and experimental main both executed'
    assert all('attn_kernel_fp8pv<8,true,16>' not in profiles[i] and 'attn_external_finalize8' not in profiles[i] for i in CONTROLS),'Control path changed'
    assert all('attn_kernel_fp8pv<' in profiles[i] for i in CONTROLS),'Retained control backend absent'
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
    assert all(ratios[i]/controls>=1.01 for i in CHANGED),'Weak or regressing affected case'
    assert summary['aggregate_ratio']>=1.005,'Insufficient weighted full-track gain'
    return summary
