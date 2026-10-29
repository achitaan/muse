"""Source/profile traffic screen only. No new kernel, API call or GPU work."""
import ast
import hashlib
import json
import re
from pathlib import Path

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'STATE.json').exists())
PARENT='93e57ab45fb71414c65b71a2b03241a62b3d6097890cb7612616e620c050aae7'
raw=(ROOT/'best/prefill_verified.py').read_bytes()
assert hashlib.sha256(raw).hexdigest()==PARENT
tree=ast.parse(raw)
functions={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
strings={t.id:ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
         and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str)
         for t in n.targets if isinstance(t,ast.Name)}
kernel=functions['kernel_fn']
calls=[n for n in ast.walk(kernel) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='_experts']
assert len(calls)==1
assert ast.unparse(calls[0].args[0])=="state['pre_expert_norm'](expert_input)"
assert sum(isinstance(n,ast.Name) and n.id=='_experts' and isinstance(n.ctx,ast.Load) for n in ast.walk(tree))==1
assert 'x.contiguous()' in ast.unparse(functions['_experts'])
cuda=strings['_pg__CUDA_SRC']
start=cuda.index('template <bool PLAN>\n__global__ void __launch_bounds__(256, MOE_GATHER_MINB) moe_gather_kernel')
end=cuda.index('// ---- routing:',start)
gather=cuda[start:end]
assert gather.count('uint4 v0 = src[0], v1 = src[1];')==1
assert 'p + (size_t)tok * 4096 + tid * 16' in gather
assert 'uint32_t outw[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};' in gather
assert '__shared__ float red[8];' in gather and len(re.findall(r'\bred\s*\[',gather))==1
assert 'rsqrt' not in gather and 'gw[' not in gather
assert gather.index('if (tok >= T)')<gather.index('uint4 v0 = src[0], v1 = src[1];')
assert 'dim3(T + (MOE_PADFILL ? 32 * MOE_PADFILL_PARTS : 0))' in strings['_pg_GLUE']
assert 'zero_centered_gamma=True' in ast.unparse(functions['_norm'])
assert '--use_fast_math' in ast.unparse(functions['_pg__build'])
profile=ROOT/'results/resumed-prefill-20261028'/PARENT/'profiles/submission.9ca79c7f676f4b63.json'
history=json.loads(profile.read_bytes())
assert history['job_id']=='9ca79c7f676f4b63'
rows=[]
for p in history['profiles']:
    case=p['case'];tokens=case['batch']*case['q_len'];report=p['report']
    matches=[line for line in report.splitlines() if 'moe_gather_kernel<' in line]
    assert len(matches)==1
    timing=re.search(r'\s(\d+(?:\.\d+)?)\s+\S+\s+([\d.]+)%\s+(\d+)\s*$',matches[0]);assert timing
    total=re.search(r'^total\s+(\d+(?:\.\d+)?)',report,re.M);assert total
    saved=2*tokens*4096*2
    rows.append(dict(index=p['index'],case=case,tokens=tokens,gather_profile_us=float(timing[1]),
                     profile_sum_us=float(total[1]),
                     candidate_logical_buffer_bytes_removed=saved,
                     nominal_HBM_byte_arithmetic_us=saved/4.8e6,
                     byte_arithmetic_is_not_measured_HBM_or_time_saved=True,
                     pre_expert_norm_profile_time=None,
                     missing_attribution='Top-kernel report alone does not identify the separate expert-input norm duration'))
report=dict(source_sha256=PARENT,profile_sha256=hashlib.sha256(profile.read_bytes()).hexdigest(),
            checked_job=history['job_id'],cpu_only=True,new_gpu_jobs=0,
            source_finding='One BF16 norm buffer write plus one immediate gather read per INPUT TOKEN; not per local route',
            unused_gather_reduction_workspace=True,zero_centered_BF16_effective_gain_required=True,
            existing_extension_uses_fast_math=True,new_norm_subnormal_handling_requires_review=True,
            all_remote_pad_rows_and_replay_contract_must_remain=True,
            source_gather_input_reads_per_token=1,source_gather_output_writes_per_local_route=1,
            rows=rows,GPU_candidate_implemented=False,
            overall_board_contribution=False,
            decision='Park GPU evaluation: no net timing bound or validated Decode transfer. If reopened, fuse norm into the existing per-token gather, remove exactly one upstream norm call and preserve the BF16 effective gain/output boundary; never normalize twice.')
(ROOT/'.tools/prefill-norm-gather-opportunity.json').write_bytes((json.dumps(report,indent=2)+'\n').encode())
print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))
for row in rows:print(json.dumps({k:row[k] for k in ('index','tokens','gather_profile_us','candidate_logical_buffer_bytes_removed','nominal_HBM_byte_arithmetic_us')}))
