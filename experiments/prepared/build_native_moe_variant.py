"""Prepare an attributed native MoE trial; never execute imported CUDA code here."""
import ast
import hashlib
import json
from pathlib import Path

root = next(p for p in Path(__file__).resolve().parents if (p / ".git").exists())
original = (root / '.tools/vjeux-moe-upstream.py').read_text()
tree = ast.parse(original)
source_node = next(n for n in tree.body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == 'CUDA_SRC' for t in n.targets))
cuda = ast.literal_eval(source_node.value)
old_cuda = cuda
helper = '''// Local change: match Marlin's BF16 accumulator/global-scale boundaries.
__device__ __forceinline__ float achita_marlin_out(float encoded_acc, float encoded_scale) {
  const float acc = __bfloat162float(__float2bfloat16_rn(encoded_acc * TWO62));
  const float gs = __bfloat162float(__float2bfloat16_rn(encoded_scale * 0x1p-62f));
  return __bfloat162float(__float2bfloat16_rn(acc * gs));
}
'''
anchor = '__device__ __forceinline__ float probe_q(float v) {'
assert cuda.count(anchor) == 1
cuda = cuda.replace(anchor, helper + anchor)
replacements = {
    'r[h] * gsc': 'achita_marlin_out(r[h], gsc)',
    'r[2 + h] * gsc': 'achita_marlin_out(r[2 + h], gsc)',
    's0 * gsc': 'achita_marlin_out(s0, gsc)',
    's1 * gsc': 'achita_marlin_out(s1, gsc)',
    '(h ? sm.y : sm.x) * gsc': 'achita_marlin_out((h ? sm.y : sm.x), gsc)',
    '(h ? sm.w : sm.z) * gsc': 'achita_marlin_out((h ? sm.w : sm.z), gsc)',
    'd4[h] * gsc': 'achita_marlin_out(d4[h], gsc)',
    'd4[2 + h] * gsc': 'achita_marlin_out(d4[2 + h], gsc)',
}
counts = {}
for a, b in replacements.items():
    counts[a] = cuda.count(a)
    assert counts[a] > 0, a
    cuda = cuda.replace(a, b)
# WGS down epilogue originally combined norm and global scale. Keep norm FP32
# after the BF16 GEMM output, using the precomputed BF16 effective norm weight.
a = 'const float w0 = (1.f + __bfloat162float(__ldg(p.normw + n0))) * gsc, w1 = (1.f + __bfloat162float(__ldg(p.normw + n0 + 8))) * gsc;'
b = 'const float w0 = __bfloat162float(__ldg(p.normw + n0)), w1 = __bfloat162float(__ldg(p.normw + n0 + 8));'
assert cuda.count(a) == 1
cuda = cuda.replace(a, b)
for a, b in [('d4[h] * w0', 'achita_marlin_out(d4[h], gsc) * w0'),
             ('d4[2 + h] * w1', 'achita_marlin_out(d4[2 + h], gsc) * w1')]:
    assert cuda.count(a) == 1
    cuda = cuda.replace(a, b)
norm_replacements = {
    '1.f + __bfloat162float(__ldg(p.normw + rbq * 16 + egg))': '__bfloat162float(__ldg(p.normw + rbq * 16 + egg))',
    '1.f + __bfloat162float(__ldg(p.normw + rbq * 16 + egg + 8))': '__bfloat162float(__ldg(p.normw + rbq * 16 + egg + 8))',
    '1.f + __bfloat162float(p.normw[n0])': '__bfloat162float(p.normw[n0])',
    '1.f + __bfloat162float(p.normw[n0 + 8])': '__bfloat162float(p.normw[n0 + 8])',
    '1.f + __uint_as_float(lnw0 << 16)': '__uint_as_float(lnw0 << 16)',
    '1.f + __uint_as_float(lnw1 << 16)': '__uint_as_float(lnw1 << 16)',
}
for a, b in norm_replacements.items():
    assert cuda.count(a) > 0, a
    cuda = cuda.replace(a, b)
assert '* gsc' not in cuda and '* w0' in cuda
assert '1.f + __bfloat162float' not in cuda

# Preserve the original Python source/comments except explicit removals and edits.
lines = original.splitlines(keepends=True)
edits = []
for n in tree.body:
    assigned = {a.id for a in n.targets if isinstance(a, ast.Name)} if isinstance(n, ast.Assign) else set()
    if 'CUDA_SRC' in assigned:
        edits.append((n.lineno - 1, n.end_lineno, 'CUDA_SRC = ' + repr(cuda) + '\n'))
    elif 'BW_SRC' in assigned or (isinstance(n, ast.FunctionDef) and n.name in
                                  {'_gen_input', '_time_graph', '_diag_bw', '_diag'}):
        edits.append((n.lineno - 1, n.end_lineno, ''))
for start, end, value in sorted(edits, reverse=True):
    lines[start:end] = [value]
result = ''.join(lines)
result = result.replace('    if DIAG:\n        _diag(state, weights)\n', '')
old = 'lut.contiguous(), weights["post_norm_w"].contiguous(),'
new = 'lut.contiguous(), (weights["post_norm_w"].float() + 1.0).to(torch.bfloat16).contiguous(),'
assert result.count(old) == 1
result = result.replace(old, new)
license_text = (root / '.tools/vjeux-LICENSE').read_text()
notice = ('# Derived from Vjeux, https://github.com/vjeux/okc-2026\n'
          '# Pinned revision f35a41ee1c13e316dabbf772830d47b23b453393; Apache-2.0.\n'
          '# Local changes: remove diagnostics; restore Marlin BF16 GEMM/global-scale\n'
          '# boundaries and BF16 effective norm weight, retaining FP32 norm arithmetic.\n'
          '# Shared scratch requires sequential replay. Full license follows.\n' +
          '\n'.join('# ' + line for line in license_text.splitlines()) + '\n')
first, rest = result.split('\n', 1)
result = first + '\n' + notice + rest
ast.parse(result)
p = root / '.tools/native-moe-prepared.py'
p.write_bytes(result.encode())
(root / '.tools/native-moe-cuda.cu').write_text(cuda)
audit = {'upstream_commit': 'f35a41ee1c13e316dabbf772830d47b23b453393',
         'upstream_sha256': hashlib.sha256(original.encode()).hexdigest(),
         'source_sha256': hashlib.sha256(result.encode()).hexdigest(),
         'rounding_sites': counts,
         'cuda_upstream_sha256': hashlib.sha256(old_cuda.encode()).hexdigest(),
         'cuda_modified_sha256': hashlib.sha256(cuda.encode()).hexdigest(),
         'gpu_status': 'unmeasured', 'contract': 'sequential shared scratch'}
(root / '.tools/native-moe-audit.json').write_text(json.dumps(audit, indent=2))
print(json.dumps(audit))
