"""CPU-only transplant draft; not authorized for evaluation before Attention wins."""
import ast
import hashlib
import json
from pathlib import Path
import sys

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'STATE.json').exists())
sys.path.insert(0,str(ROOT/'experiments/prepared'))
from build_attention_skip_identity_scale import transform
sys.path.insert(0,str(ROOT/'scripts'))
from verify_attention_precision import verify

folder = ROOT/'results/decoder-unused-rope-prune-20261029'
raw = (folder/'submission.py').read_bytes()
meta = json.loads((folder/'cpu-source-audit.json').read_bytes())
assert hashlib.sha256(raw).hexdigest() == meta['source_sha256']
assert meta['source_sha256'].startswith('307dd791')
assert meta['all_runtime_functions_exact'] and meta['all_native_attention_and_fallback_source_exact']
tree=ast.parse(raw)
assignment=next(n for n in tree.body if isinstance(n,ast.Assign)
                and any(isinstance(t,ast.Name) and t.id=='_native_attn_CUDA_SRC' for t in n.targets))
old=ast.literal_eval(assignment.value)
new,edits=transform(old,diagnostic=False)
assert len(edits)==3
restored=new
for edit in reversed(edits):
    start=edit['offset'];assert restored[start:start+len(edit['new'])]==edit['new']
    restored=restored[:start]+edit['old']+restored[start+len(edit['new']):]
assert restored==old
lines=raw.splitlines(keepends=True);node=assignment.value
start=sum(map(len,lines[:node.lineno-1]))+node.col_offset
end=sum(map(len,lines[:node.end_lineno-1]))+node.end_col_offset
result=raw[:start]+repr(new).encode()+raw[end:]
check=ast.parse(result)
current=next(n for n in check.body if isinstance(n,ast.Assign)
             and any(isinstance(t,ast.Name) and t.id=='_native_attn_CUDA_SRC' for t in n.targets))
current.value=assignment.value
assert ast.dump(check,include_attributes=False)==ast.dump(tree,include_attributes=False)
assert result[:start]+raw[start:end]+result[start+len(repr(new).encode()):]==raw
assert len(result)<1_000_000
compile(result,'<decoder-skipscale-cpu-draft>','exec')
verify(result,'_native_attn_')
report=dict(parent_package_sha256=meta['source_sha256'],source_sha256=hashlib.sha256(result).hexdigest(),
            bytes=len(result),source_headroom=1_000_000-len(result),
            original_D9_sha256='d9fe7558624a1f070bbacb493b94302e90ff9bf77cdd8d4239f9d7a83ab39988',
            parent_attention_cuda_sha256=hashlib.sha256(old.encode()).hexdigest(),
            attention_cuda_sha256=hashlib.sha256(new.encode()).hexdigest(),
            arithmetic_edits=edits,all_other_package_Python_AST_exact=True,
            inverse_attention_CUDA_and_parent_package_bytes_exact=True,
            original_FP16_PV_preserved=True,standalone_tied_FP8_PV_not_transferred=True,
            BF16_boundary_guard_passed=True,cpu_only=True,GPU_submission_ready=False,
            active_decode_candidate_changed=False,gpu_verified=False,
            next_condition='Attention standalone source-bound material-gain gate first; then exact resource/actual-execution binding and full whole-layer correctness/weighted latency evaluation. No GPU job from this draft alone.')
(ROOT/'.tools/decoder-skipscale-draft.py').write_bytes(result)
(ROOT/'.tools/decoder-skipscale-draft-audit.json').write_bytes((json.dumps(report,indent=2)+'\n').encode())
print(json.dumps({k:v for k,v in report.items() if k!='arithmetic_edits'},indent=2))
