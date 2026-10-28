"""Prepare a Prefill grouped-MoE hypothesis without launching any GPU work."""
import ast
import hashlib
import json
import re
from pathlib import Path

root = next(p for p in Path(__file__).resolve().parents if (p / ".git").exists())
original = (root / '.tools/vjeux-prefill-upstream.py').read_text()
tree = ast.parse(original)
selected_functions = {'_fast_scales_ok', '_codes', '_repack', '_build', '_max_tiles', '_moe_forward'}
selected = []
for n in tree.body:
    if isinstance(n, ast.FunctionDef) and n.name in selected_functions:
        selected.append(n)
    elif isinstance(n, (ast.Assign, ast.AnnAssign)):
        # Module constants contain the native payloads and fixed launch knobs;
        # no setup or diagnostics are invoked by these assignments.
        selected.append(n)
# Keep only constants referenced by the extracted functions and their values.
def bound_names(n):
    return {x.id for target in (n.targets if isinstance(n, ast.Assign) else [])
            for x in ast.walk(target) if isinstance(x, ast.Name)}
needed = {x.id for n in selected if isinstance(n, ast.FunctionDef)
          for x in ast.walk(n) if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Load)}
needed.add('MOE_SMEM_SM90')  # referenced by the local setup wrapper below
while True:
    expanded = needed | {x.id for n in selected if isinstance(n, ast.Assign) and bound_names(n) & needed
                         for x in ast.walk(n.value) if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Load)}
    if expanded == needed: break
    needed = expanded
selected = [n for n in selected if isinstance(n, ast.FunctionDef) or bound_names(n) & needed]
mod = ast.Module(body=selected, type_ignores=[])
cuda_node = next(n for n in selected if isinstance(n, ast.Assign)
                 and any(isinstance(a, ast.Name) and a.id == '_CUDA_SRC' for a in n.targets))
cuda = ast.literal_eval(cuda_node.value)
helper = '''// Local change: preserve Marlin's BF16 GEMM/global-scale boundaries.
DEVI float achita_marlin_out(float encoded_acc, float encoded_scale) {
  const float acc = __bfloat162float(__float2bfloat16_rn(encoded_acc * 128.f));
  const float gs = __bfloat162float(__float2bfloat16_rn(encoded_scale * 0x1p-7f));
  return __bfloat162float(__float2bfloat16_rn(acc * gs));
}
'''
anchor = 'template <int N, bool GATED>\nDEVI void consumer_tile'
assert cuda.count(anchor) == 1
cuda = cuda.replace(anchor, helper + anchor)
cuda, rounding_sites = re.subn(r'(acc\[[^\]]+\]) \* cs\b', r'achita_marlin_out(\1, cs)', cuda)
assert rounding_sites == 14, rounding_sites
assert not re.search(r'acc\[[^\]]+\] \* cs\b', cuda)
# Input to replacement _experts is already normalized by SGLang. Copy raw
# BF16 bits, including subnormals, into the native tile layout. Retain the
# shared-memory barrier publishing route locations to every thread.
start = '  float f[16];\n'
end = '  const int stg = tid >> 2, ch0 = 2 * (tid & 3);'
assert cuda.count(start) == 1 and cuda.count(end) == 1
i, j = cuda.index(start), cuda.index(end)
assert i < j
cuda = cuda[:i] + '''  uint32_t outw[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
  __syncthreads();  // route metadata published by threads 0..7
''' + cuda[j:]
cuda_node.value = ast.Constant(cuda)

# Namespace every module binding, including declarations; preserve literal CUDA.
symbols = set()
for n in selected:
    if isinstance(n, ast.FunctionDef): symbols.add(n.name)
    elif isinstance(n, ast.Assign):
        for target in n.targets:
            symbols.update(x.id for x in ast.walk(target) if isinstance(x, ast.Name))
rename = {name: '_pg_' + name for name in symbols}
class Rename(ast.NodeTransformer):
    def visit_Name(self, node):
        node.id = rename.get(node.id, node.id)
        return node
    def visit_FunctionDef(self, node):
        node.name = rename.get(node.name, node.name)
        return self.generic_visit(node)
    def visit_Global(self, node):
        node.names = [rename.get(x, x) for x in node.names]
        return node
mod = Rename().visit(mod)
ast.fix_missing_locations(mod)
native = ast.unparse(mod)

baseline = (root / '.tools/prefill-starter/baseline.py').read_text()
bt = ast.parse(baseline)
bank = next(n for n in bt.body if isinstance(n, ast.FunctionDef) and n.name == '_marlin_bank')
lines = baseline.splitlines(keepends=True)
lines[bank.lineno - 1:bank.end_lineno] = ['def _marlin_bank(weights):\n    return None\n']
baseline = ''.join(lines)

wrapper = '''

@triton.jit
def _pg_route_metadata(IDS, LOCAL, POS, COUNTS, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    e = tl.load(IDS + i, mask=i < N, other=-1)
    local = (e >= 0) & (e < 32) & (i < N)
    p = tl.atomic_add(COUNTS + tl.minimum(tl.maximum(e, 0), 31), 1, mask=local)
    tl.store(LOCAL + i, tl.where(local, e, -1), mask=i < N)
    tl.store(POS + i, tl.where(local, p, -1), mask=i < N)

@triton.jit
def _pg_combine(Y, ROW, LOCAL, WEIGHT, NORM, OUT, H: tl.constexpr):
    t = tl.program_id(0)
    dim = tl.arange(0, H)
    norm = tl.load(NORM + dim).to(tl.float32)
    result = tl.zeros((H,), tl.float32)
    for k in tl.static_range(8):
        e = tl.load(LOCAL + t * 8 + k)
        if e >= 0:
            row = tl.load(ROW + t * 8 + k)
            y = tl.load(Y + row.to(tl.int64) * H + dim).to(tl.float32)
            z = y * norm
            inv = tl.rsqrt(tl.sum(z * z, 0) / H + 1e-8)
            w = tl.load(WEIGHT + t * 8 + k)
            result += (z * inv) * w
    tl.store(OUT + t.to(tl.int64) * H + dim, result.to(tl.bfloat16))

_pg_original_prepare = prepare
def prepare(weights):
    state = _pg_original_prepare(weights)
    dev = weights['w13'].device
    mod = _pg__build(_pg__fast_scales_ok(weights))
    smem = _pg_MOE_SMEM_SM90
    mod.moe_init(smem)
    state['pg'] = {
        'mod': mod, 'smem': smem,
        'nsm': torch.cuda.get_device_properties(dev).multi_processor_count,
        'w13r': _pg__repack(weights['w13'], weights['w13_sf'], True),
        'w2r': _pg__repack(weights['w2'], weights['w2_sf'], False),
        'w13_gs': weights['w13_gs'].contiguous(),
        'w2_gs': weights['w2_gs'].contiguous(),
        'g_pre_expert_norm': torch.zeros((4096,), dtype=torch.bfloat16, device=dev),
        'g_post_expert_norm': state['post_expert_w'],
        'no_zero': torch.empty((0,), dtype=torch.int32, device=dev),
        'no_zero_f': torch.empty((0,), dtype=torch.float32, device=dev),
        'no_zero_b': torch.empty((0,), dtype=torch.bfloat16, device=dev),
        'moe_dbg': torch.empty((0,), dtype=torch.int64, device=dev), 'moe_flags': 0}
    return state

def _experts(x, topk_weights, topk_ids, state):
    T = x.shape[0]
    ids = topk_ids.to(torch.int32).contiguous()
    weights = topk_weights.to(torch.float32).contiguous()
    local = torch.empty_like(ids)
    pos = torch.empty_like(ids)
    counts = torch.zeros((32,), dtype=torch.int32, device=x.device)
    _pg_route_metadata[(triton.cdiv(T * 8, 256),)](ids, local, pos, counts, T * 8, 256)
    ex = _pg__moe_forward(state['pg'], x.contiguous(), local, pos, counts, T)
    out = torch.empty((T, 4096), dtype=torch.bfloat16, device=x.device)
    _pg_combine[(T,)](ex['y'], ex['rrow'], local, weights, state['post_expert_w'], out, 4096, num_warps=8)
    return out
'''
license_text = (root / '.tools/vjeux-LICENSE').read_text()
notice = ('# Grouped MoE derived from Vjeux, okc-2026, Apache-2.0, revision\n'
          '# f35a41ee1c13e316dabbf772830d47b23b453393.\n'
          '# Local changes: component extraction, metadata route builder, SG BF16\n'
          '# GEMM boundaries, copy-only gather, FP32 norm/weighted-sum integration.\n' +
          '\n'.join('# ' + x for x in license_text.splitlines()) + '\n')
combined = ('# okc: problem=muse_prefill_layer\n' + notice + '\n' +
            'import hashlib\nimport torch\nfrom torch.utils.cpp_extension import load_inline\n'
            'import triton\nimport triton.language as tl\n' + native + '\n' + baseline + wrapper)
compile(combined, '<prefill-grouped-draft>', 'exec')
# The full layer's router, normalization and residual implementation is retained.
assert 'F.linear(router_input, state["router"])' in combined
assert 'logits += torch.mm(ffn_input, state["router_lo_t"], out_dtype=torch.float32)' in combined
out = root / '.tools/prefill-grouped-variant.py'
out.write_bytes(combined.encode())
audit = {'source_sha256': hashlib.sha256(combined.encode()).hexdigest(),
         'upstream_commit': 'f35a41ee1c13e316dabbf772830d47b23b453393',
         'bf16_boundary_sites': rounding_sites,
         'gather_normalization': 'disabled: input normalized by original SGLang module',
         'norm_intermediates': 'FP32; effective norm weight BF16 from SGLang',
         'router_attention_residuals': 'original official baseline unchanged',
         'gpu_status': 'unmeasured', 'packing_inverse': 'pending'}
(root / '.tools/prefill-grouped-audit.json').write_text(json.dumps(audit, indent=2))
print(json.dumps(audit))
