"""Prepare guarded FP32 tanh SiLU and a prepare-only public-data GPU audit."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'STATE.json').exists())
PARENT = '331148716e8d1ba92b45cd29725be997d44b3a5235af74d75770bb9444100d50'
raw = (ROOT / 'results/resumed-moe-phase4-20261028' / PARENT / 'submission.py').read_bytes()
assert hashlib.sha256(raw).hexdigest() == PARENT
source = raw.decode()
tree = ast.parse(source)
nodes = {x.id: n for n in tree.body if isinstance(n, ast.Assign) for x in n.targets if isinstance(x, ast.Name)}
cuda_node, cpp_node = nodes['CUDA_SRC'], nodes['CPP_SRC']
cuda = ast.literal_eval(cuda_node.value)
cpp = ast.literal_eval(cpp_node.value)
old_expression = 'SILU_DIV(gt, 1.f + __expf(-gt))'
assert cuda.count(old_expression) == 5
cuda = cuda.replace(old_expression, 'achita_silu_guarded(gt)')
anchor = '__device__ __forceinline__ float probe_q(float v) {'
helper = '''// FP32 SiLU identity; retain exp/div in the negative tail to avoid cancellation.
// The final BF16 SwiGLU store and both Marlin BF16 boundaries remain unchanged.
__device__ __forceinline__ float achita_silu_guarded(float x) {
  if (x < -4.f) return SILU_DIV(x, 1.f + __expf(-x));
  float t;
  asm("tanh.approx.f32 %0, %1;" : "=f"(t) : "f"(.5f * x));
  return .5f * x * (1.f + t);
}
'''
assert cuda.count(anchor) == 1
cuda = cuda.replace(anchor, helper + anchor)
extra_cuda = '''
// Prepare-only diagnostics on all finite BF16 gates; no challenge data is inspected.
__global__ void achita_silu_audit_kernel(const float* x, float* previous, float* next, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    const float g = x[i];
    previous[i] = SILU_DIV(g, 1.f + __expf(-g));
    next[i] = okc::achita_silu_guarded(g);
  }
}
void okc_silu_audit(torch::Tensor x, torch::Tensor previous, torch::Tensor next) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kFloat && x.is_contiguous());
  TORCH_CHECK(previous.sizes() == x.sizes() && next.sizes() == x.sizes());
  achita_silu_audit_kernel<<<(x.numel() + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      x.data_ptr<float>(), previous.data_ptr<float>(), next.data_ptr<float>(), x.numel());
}
'''
cuda += extra_cuda
new_cpp = cpp + '\nvoid okc_silu_audit(torch::Tensor x, torch::Tensor previous, torch::Tensor next);\n'
old_cuda_literal, old_cpp_literal = ast.get_source_segment(source, cuda_node.value), ast.get_source_segment(source, cpp_node.value)
output = source.replace(old_cuda_literal, repr(cuda), 1).replace(old_cpp_literal, repr(new_cpp), 1)
ext = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_ext')
load = next(n for n in ast.walk(ext) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'load_inline')
fn_list = next(k.value for k in load.keywords if k.arg == 'functions')
assert ast.literal_eval(fn_list) == ['okc_run', 'okc_set_pdl']
old_functions = ast.get_source_segment(source, fn_list)
new_functions = repr(['okc_run', 'okc_set_pdl', 'okc_silu_audit'])
assert output.count(old_functions) == 1
output = output.replace(old_functions, new_functions)
old_tag, new_tag = 'okc_moe_w4a16_v216x_pair1dq8_', 'okc_moe_w4a16_v216x_pair1dq8_silu_'
assert output.count(old_tag) == 1
output = output.replace(old_tag, new_tag)
wrapper = '''

_achita_original_prepare = prepare

def prepare(weights):
    state = _achita_original_prepare(weights)
    dev = weights['w13'].device
    # All inputs here are public synthetic values, independent of timed data.
    bits = torch.arange(65536, dtype=torch.int32, device=dev)
    bits = bits[(bits & 0x7f80) != 0x7f80]
    gates = bits.to(torch.int16).view(torch.bfloat16).float().contiguous()
    old, new = torch.empty_like(gates), torch.empty_like(gates)
    mod = _ext(tuple(_main_defines(state['sfx'])) + state['sfx'])
    mod.okc_silu_audit(gates, old, new)
    g64 = gates.double()
    z = torch.exp(-g64.abs())
    wanted = g64 * torch.where(g64 >= 0, 1 / (1 + z), z / (1 + z))
    denom = torch.maximum(wanted.abs(), torch.full_like(wanted, 1e-5))
    scaled_error = ((new.double() - wanted).abs() / denom).max().item()
    original_error = ((old.double() - wanted).abs() / denom).max().item()
    finite = bool(torch.isfinite(new).all().item())
    mask = gates.abs() <= 32
    a, b = new[mask], wanted[mask]
    up = (1 + (torch.arange(a.numel(), device=dev) % 257).float() / 32).bfloat16().float()
    observed = (a * up).bfloat16().float()
    target = (b * up.double()).bfloat16().float()
    bf16_l2 = (torch.linalg.vector_norm(observed - target) / torch.linalg.vector_norm(target)).item()
    torch.cuda.synchronize()
    print('MOE_SILU_AUDIT', dict(count=gates.numel(), all_finite=finite,
          scaled_error=scaled_error, original_scaled_error=original_error, bf16_stage_l2=bf16_l2,
          cutoff=-4, oracle='stable FP64 SiLU on all finite BF16 gates'))
    if not finite or scaled_error >= 2e-4 or bf16_l2 >= .001:
        raise RuntimeError('Guarded tanh SiLU failed public full-domain GPU activation audit')
    return state
'''
output += wrapper
compile(output, '<moe-guarded-silu>', 'exec')
restored = cuda[:-len(extra_cuda)].replace(helper, '').replace('achita_silu_guarded(gt)', old_expression)
assert restored == ast.literal_eval(cuda_node.value)
assert output[:-len(wrapper)].replace(repr(cuda), old_cuda_literal, 1).replace(repr(new_cpp), old_cpp_literal, 1).replace(new_functions, old_functions).replace(new_tag, old_tag) == source
out = output.encode()
(ROOT / '.tools/moe-silu.py').write_bytes(out)
report = dict(parent_sha256=PARENT, source_sha256=hashlib.sha256(out).hexdigest(),
              silu_sites=5, negative_tail_cutoff=-4, exact_source_inverse=True,
              kernel_fn_and_all_nonactivation_math_unchanged=True,
              gpu_diagnostic='all65280finiteBF16gates with stableFP64oracle, independent of timed data',
              gpu_status='unmeasured')
(ROOT / '.tools/moe-silu-source-audit.json').write_bytes(json.dumps(report, indent=2).encode())
print(json.dumps(report))
