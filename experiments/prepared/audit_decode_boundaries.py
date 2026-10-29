"""Source-bound Decode dataflow map; no imports of submission or GPU work."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'STATE.json').exists())
PARENT = 'd9fe7558624a1f070bbacb493b94302e90ff9bf77cdd8d4239f9d7a83ab39988'
raw = (ROOT / 'best/decoder_verified.py').read_bytes()
assert hashlib.sha256(raw).hexdigest() == PARENT
tree = ast.parse(raw)
functions = {}
constants = {}
for node in tree.body:
    if isinstance(node, ast.FunctionDef):
        functions[node.name] = node
    if isinstance(node, ast.Assign):
        try:
            value = ast.literal_eval(node.value)
        except (ValueError, TypeError):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                constants[target.id] = value
assert constants['ROUTER_FP32_MAX_TOKENS'] == 8
assert constants['SLICED_UNPOOL_MAX_TOKENS'] == 8
assert constants['HIDDEN'] == 8192
slice_assignment = next(n for n in tree.body if isinstance(n, ast.Assign)
                        and any(isinstance(q, ast.Name) and q.id == 'UNPOOL_SLICE' for q in n.targets))
assert ast.unparse(slice_assignment.value) == 'HIDDEN // 8'
kernel = functions['kernel_fn']
source = ast.get_source_segment(raw.decode(), kernel)
expected_calls = ('_achita_dense_project', '_native_attn_kernel_fn', '_fpr_bfp_fused_pooler_router',
                  '_rf_route', '_achita_wide_router', '_pn_expert_inputs', '_experts', '_sfn_boundary')
calls = [ast.unparse(n.func) for n in ast.walk(kernel) if isinstance(n, ast.Call)]
assert all(name in calls for name in expected_calls)
assert "if 8 < tokens <= 32 and state['fpr'] is not None:" in source
assert source.index('side_stream.wait_stream(main_stream)') < source.index("'pooler'")
assert source.index('main_stream.wait_stream(side_stream)') < source.index('_pn_expert_inputs(')
assert 'enable_fp_fusion=False' in ast.get_source_segment(raw.decode(), functions['_pn_expert_inputs'])
assert 'enable_fp_fusion=False' in ast.get_source_segment(raw.decode(), functions['_sfn_boundary'])
helpers = ('kernel_fn', '_pn_expert_inputs', '_pn_inputs_kernel', '_sfn_boundary', '_sfn_kernel',
           '_achita_wide_router', '_experts')
bindings = {name: dict(first_line=functions[name].lineno,
                       ast_sha256=hashlib.sha256(ast.dump(functions[name], include_attributes=False).encode()).hexdigest())
            for name in helpers}
profiles = ROOT / 'results/retained-load-opportunities-20261029/loads.json'
data = json.loads(profiles.read_bytes())
study = data['studies']['decode']
assert study['source_sha256'] == PARENT and study['job'] == '9560c915e85f4f50'
report = dict(source_sha256=PARENT, cpu_only=True, new_gpu_jobs=0,
              retained_load_table_sha256=hashlib.sha256(profiles.read_bytes()).hexdigest(),
              final_function_bindings=bindings,
              router_fp32_max_tokens=8, fused_pooler_router_tokens=[9, 32],
              sliced_zero_tail_unpooler_max_tokens=8,
              sliced_unpooler_columns=1024, full_ffn_norm_domain=8192,
              runtime_dependency=['QKV', 'native attention/cache/norm/gate', 'O',
                                  'attention residual/norm', 'pooler + router/routing',
                                  'pre-expert norm/local-ID mapping', 'MoE', 'unpooler',
                                  'FFN residual/norm'],
              concurrency='Outside eligible9..32 fusion, pooler runs on main stream while router/routing run on side stream; join precedes expert preprocessing',
              metadata_reads_are_not_timeline_proof=True,
              accepted_runtime_candidate=None,
              rejected_shortcuts=['Treat source launch order as a measured critical path',
                                  'Fuse concurrent router/pooler by serializing their work',
                                  'Normalize GEMM partial output tiles independently',
                                  'Re-add the retained expert-input or small zero-tail fusion',
                                  'Transfer rejected MoE factor cache'],
              next_evidence='A distinct source mechanism and conservative recoverable whole-layer time; preserve source/profile regime identity and all precision/fallback boundaries')
out = ROOT / 'results/decode-boundary-audit-20261029'
out.mkdir(exist_ok=True)
target = out / 'audit.json'
encoded = (json.dumps(report, indent=2) + '\n').encode()
assert not target.exists() or target.read_bytes() == encoded, 'Frozen audit collision'
target.write_bytes(encoded)
print(json.dumps(report, indent=2))

