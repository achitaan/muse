"""Protect BF16 materializations in standalone and embedded native attention.

The independent numerical evidence is in
experiments/prepared/audit_attention_bf16_boundaries_cpu.py. This source guard
prevents silently restoring the inherited identity-output/FP32 QK variant.
"""
import ast
from pathlib import Path
import sys


def verify(source, prefix=''):
    tree = ast.parse(source)
    name = prefix + 'CUDA_SRC'
    cuda = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))
    assert '__bfloat162float(__float2bfloat16_rn(x))' in cuda
    assert cuda.count('x[e] = bf16_round(x[e] * r);') == 3
    assert cuda.count('x[2 * e] = bf16_round(a0 * c - a1 * sn);') == 3
    assert cuda.count('x[2 * e + 1] = bf16_round(a0 * sn + a1 * c);') == 3
    assert cuda.count('attn_out_bf16(') == 16 and 'pr8(' not in cuda
    assert 'float attn_out_bf16(float x) {\n  return bf16_round(x);\n}' in cuda
    fallback = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == prefix + '_fallback')
    text = ast.unparse(fallback)
    assert text.count('.to(torch.bfloat16).float()') == 2
    cast = ast.parse('q, k = q.to(torch.bfloat16), k.to(torch.bfloat16)').body[0]
    assert any(ast.dump(n) == ast.dump(cast) for n in fallback.body)
    assert "cfg['sm_scale'] * cfg['qk_scale_factor']" in text
    print('Native attention retains BF16 norm, RoPE and output boundaries in all finalizers and fallback.')


if __name__ == '__main__':
    path = Path(sys.argv[1])
    verify(path.read_bytes(), '_native_attn_' if len(sys.argv) > 2 and sys.argv[2] == 'decode' else '')
