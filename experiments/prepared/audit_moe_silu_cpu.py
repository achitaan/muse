"""CPU screen of guarded tanh SiLU; does not emulate PTX hardware approximation."""
import ast
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'STATE.json').exists())
candidate_source = (ROOT / '.tools/moe-silu.py').read_bytes()
source_audit = json.loads((ROOT / '.tools/moe-silu-source-audit.json').read_bytes())
assert hashlib.sha256(candidate_source).hexdigest() == source_audit['source_sha256']
tree = ast.parse(candidate_source)
cuda = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
            and any(isinstance(x, ast.Name) and x.id == 'CUDA_SRC' for x in n.targets))
assert 'if (x < -4.f) return SILU_DIV(x, 1.f + __expf(-x));' in cuda
assert 'return .5f * x * (1.f + t);' in cuda
assert cuda.count('achita_silu_guarded(gt)') == 5
torch.set_num_threads(2)
bits = np.arange(65536, dtype=np.uint16)
bits = bits[(bits & 0x7f80) != 0x7f80]
gate = torch.from_numpy(bits.view(np.int16).copy()).view(torch.bfloat16).float()


def oracle(x):
    # Stable FP64 sigmoid; exp's argument is never positive.
    x = x.double()
    t = torch.exp(-x.abs())
    return x * torch.where(x >= 0, 1 / (1 + t), t / (1 + t))


def candidate(x):
    # Negative tail retains the existing exp/div evaluation to avoid tanh cancellation.
    return torch.where(x < -4, x / (1 + torch.exp(-x)), .5 * x * (1 + torch.tanh(.5 * x)))


expected = oracle(gate)
new = candidate(gate)
assert torch.isfinite(new).all()
scaled_error = ((new.double() - expected).abs() / torch.maximum(expected.abs(), torch.tensor(1e-5))).max().item()
assert scaled_error < 2e-5
torch.manual_seed(20261028)
records = []
for sigma in (.25, 1., 4., 16.):
    for mean in (-4., 0., 4.):
        x = (mean + sigma * torch.randn(32, 4096)).bfloat16().float()
        up = (torch.randn_like(x) * 4).bfloat16().float()
        wanted = (oracle(x) * up.double()).bfloat16().float()
        got = (candidate(x) * up).bfloat16().float()
        # Fixed random signed linear map tests cancellation and per-row RMS sensitivity.
        w = torch.randn(4096, 128).bfloat16().float() / 64
        a, b = wanted @ w, got @ w
        a *= torch.rsqrt((a * a).mean(-1, keepdim=True) + 1e-8)
        b *= torch.rsqrt((b * b).mean(-1, keepdim=True) + 1e-8)
        row_error = torch.linalg.vector_norm(a - b, dim=1) / torch.linalg.vector_norm(a, dim=1)
        records.append(dict(mean=mean, sigma=sigma,
                            bf16_intermediate_relative_l2=float(torch.linalg.vector_norm(wanted - got) / torch.linalg.vector_norm(wanted)),
                            normalized_linear_max_row_l2=float(row_error.max()),
                            exact_tail_fraction=float((x < -4).float().mean())))
assert max(r['normalized_linear_max_row_l2'] for r in records) < .001
report = dict(source_sha256=source_audit['source_sha256'], seed=20261028, finite_bf16_gate_values=len(gate), negative_tail_cutoff=-4,
              max_fp32_cpu_error_scaled_by_max_reference_or_1e_5=scaled_error,
              cases=records, hardware_tanh_error_bound_relative=2**-11,
              hardware_error_bound_source='https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-tanh',
              cpu_is_not_ptx_emulator=True, gpu_correctness='unmeasured', speedup='unmeasured',
              required_before_retention=['actual GPU full-domain activation audit', 'full official 5/6/integrity', 'material speedup'])
(ROOT / '.tools/moe-silu-cpu-audit.json').write_bytes(json.dumps(report, indent=2).encode())
print(json.dumps(report))
