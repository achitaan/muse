"""Compare implicit padding with the pinned SGLang boundary arithmetic on CPU."""
import ast
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

rng = np.random.default_rng(20261028)
H, SLICE = 8192, 1024


def bf16(x):
    return torch.from_numpy(np.asarray(x, dtype=np.float32)).to(torch.bfloat16).float().numpy()


def boundary_reference(branch, stream, post_weight, pre_weight, alpha, beta):
    # Pinned SGLang norm_residual.cuh, single-CTA path: BF16 scale-first
    # product, BF16 normalized branch, FP32 residual and pre-norm input.
    post = bf16(branch * post_weight)
    inv = np.float32(1.) / np.sqrt((post * post).sum(-1, keepdims=True, dtype=np.float32) / np.float32(H) + np.float32(1e-8))
    normalized = bf16(post * inv)
    updated = alpha * stream + beta * normalized
    pre_inv = np.float32(1.) / np.sqrt((updated * updated).sum(-1, keepdims=True, dtype=np.float32) / np.float32(H) + np.float32(1e-5))
    return bf16((updated * pre_inv) * pre_weight), updated


def implicit_model(cropped, stream, post_weight, pre_weight, alpha, beta):
    # Independent cropped-domain calculation: zeros contribute nothing to
    # the numerator, but the norm denominator remains H, never SLICE.
    post = bf16(cropped * post_weight[:SLICE])
    inv = np.float32(1.) / np.sqrt((post * post).sum(-1, keepdims=True, dtype=np.float32) / np.float32(H) + np.float32(1e-8))
    normalized = bf16(post * inv)
    updated = (alpha * stream).copy()
    updated[:, :SLICE] += beta[:SLICE] * normalized
    pre_inv = np.float32(1.) / np.sqrt((updated * updated).sum(-1, keepdims=True, dtype=np.float32) / np.float32(H) + np.float32(1e-5))
    return bf16((updated * pre_inv) * pre_weight), updated


cases = 0
max_stream_error = max_output_error = 0.
for rows in (1, 3, 8):
    for kind in ('zero', 'normal', 'outlier', 'small', 'zero_stream', 'constant'):
        cropped = rng.normal(size=(rows, SLICE)).astype(np.float32)
        stream = rng.normal(size=(rows, H)).astype(np.float32)
        if kind == 'zero': cropped.fill(0)
        if kind == 'small': cropped *= 1e-6
        if kind == 'outlier': cropped[:, ::71] *= 64
        if kind == 'zero_stream': stream.fill(0)
        if kind == 'constant': cropped.fill(.125)
        cropped = bf16(cropped)
        post_weight = bf16(1 + bf16(rng.normal(0, .3, H)))
        pre_weight = bf16(1 + bf16(rng.normal(0, .1, H)))
        gate = rng.normal(0, 3, H).astype(np.float32) / np.float32(.3)
        beta = (1 / (1 + np.exp(-gate))).astype(np.float32)
        alpha = np.sqrt(np.maximum((1 / (1 + np.exp(gate))) * (1 + beta), np.float32(1e-3))).astype(np.float32)
        full = np.zeros((rows, H), dtype=np.float32)
        full[:, :SLICE] = cropped
        reference, expected_stream = boundary_reference(full, stream, post_weight, pre_weight, alpha, beta)
        got, got_stream = implicit_model(cropped, stream, post_weight, pre_weight, alpha, beta)
        stream_error = np.linalg.norm(got_stream - expected_stream) / max(np.linalg.norm(expected_stream), 1e-30)
        output_error = np.linalg.norm(got - reference) / max(np.linalg.norm(reference), 1e-30)
        assert np.isfinite(got).all() and np.isfinite(got_stream).all()
        assert stream_error < 1e-5 and output_error < 0.001
        max_stream_error = max(max_stream_error, float(stream_error))
        max_output_error = max(max_output_error, float(output_error))
        cases += 1
# Analytic constant case catches the erroneous 1024-wide RMSNorm denominator.
_, stream = implicit_model(np.ones((1, SLICE), dtype=np.float32), np.zeros((1, H), dtype=np.float32),
                           np.ones(H, dtype=np.float32), np.ones(H, dtype=np.float32),
                           np.ones(H, dtype=np.float32), np.ones(H, dtype=np.float32))
assert np.array_equal(stream[0, :SLICE], np.full(SLICE, 2.828125, dtype=np.float32))
assert not stream[0, SLICE:].any()
source = Path('.tools/decoder-sliced-ffn-variant.py').read_bytes()
tree = ast.parse(source)
fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_sfn_kernel')
text = ast.unparse(fn)
assert 'col < Slice' in text and '/ H + PostEps' in text
assert 'updated.to(' not in text
record = {'source_sha256': hashlib.sha256(source).hexdigest(), 'seed': 20261028,
          'manufactured_cases': cases, 'normalization_domain': H, 'stored_branch_columns': SLICE,
          'max_stream_relative_l2': max_stream_error, 'max_output_relative_l2': max_output_error,
          'analytic_constant_expected_stream_prefix': 2.828125, 'zero_tail_preserved': True,
          'bf16_product_and_branch_boundaries_preserved': True, 'pre_norm_input_fp32': True,
          'gpu_status': 'unmeasured'}
Path('.tools/decoder-sliced-ffn-cpu-audit.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
print(json.dumps(record))
