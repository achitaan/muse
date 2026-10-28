"""Prepare a BF16x2 epilogue trial, preserving Marlin's rounding boundaries."""
import ast
import hashlib
import json
import re
from pathlib import Path

root = next(p for p in Path(__file__).resolve().parents if (p / '.git').exists())
sha = 'dd6049cbf7a2a41666f2d9e46010f81e0dd22c0840881bc5f187ac5f3ba1d3a3'
raw = (root / 'results/resumed-moe-phase3-20261028' / sha / 'submission.py').read_bytes()
assert hashlib.sha256(raw).hexdigest() == sha
source = raw.decode('utf-8')
tree = ast.parse(source)
node = next(n for n in tree.body if isinstance(n, ast.Assign)
            and any(isinstance(a, ast.Name) and a.id == 'CUDA_SRC' for a in n.targets))
cuda = ast.literal_eval(node.value)
original_cuda = cuda
old = '''__device__ __forceinline__ float achita_marlin_out(float encoded_acc, float encoded_scale) {
  const float acc = __bfloat162float(__float2bfloat16_rn(encoded_acc * TWO62));
  const float gs = __bfloat162float(__float2bfloat16_rn(encoded_scale * 0x1p-62f));
  return __bfloat162float(__float2bfloat16_rn(acc * gs));
}'''
new = '''// Local change: pair independent outputs, retaining both BF16 rounding boundaries.
// CUDA Math API: __hmul2_rn is RN-even and prevents contraction with additions.
__device__ __forceinline__ float2 achita_marlin_out2(float encoded_acc0, float encoded_acc1, float encoded_scale) {
  const __nv_bfloat162 acc = __floats2bfloat162_rn(encoded_acc0 * TWO62, encoded_acc1 * TWO62);
  const __nv_bfloat16 gs = __float2bfloat16_rn(encoded_scale * 0x1p-62f);
  return __bfloat1622float2(__hmul2_rn(acc, __bfloat162bfloat162(gs)));
}'''
assert cuda.count(old) == 1
cuda = cuda.replace(old, new)
counter = [0]
replacements = []


# Pair values that already share one scalar declaration and scale.
pattern = (r'(?P<indent> +)(?P<qual>const )?float (?P<v0>gt|v0|v0f) = '
           r'achita_marlin_out\((?P<a>[^\n]+?), gsc\)(?P<tail0>[^,;]*), '
           r'(?P<v1>up|v1|v1f) = achita_marlin_out\((?P<b>[^\n]+?), gsc\)(?P<tail1>[^;]*);')


def with_indent(match):
    # An absent optional group is the empty qualifier.
    d = match.groupdict()
    number = counter[0]
    counter[0] += 1
    p = f'achita_pair_{number}'
    paired = (f"{d['indent']}const float2 {p} = achita_marlin_out2({d['a']}, {d['b']}, gsc);\n"
              f"{d['indent']}{d['qual'] or ''}float {d['v0']} = {p}.x{d['tail0']}, {d['v1']} = {p}.y{d['tail1']};")
    replacements.append((paired, match.group(0)))
    return paired


cuda, pairs = re.subn(pattern, with_indent, cuda)
assert pairs == 10, pairs
old_stores = '''        st_keep_f32(p.ybuf + ((size_t)(jst + tk) * HD + n0), achita_marlin_out(d4[h], gsc) * w0);
        st_keep_f32(p.ybuf + ((size_t)(jst + tk) * HD + n0 + 8), achita_marlin_out(d4[2 + h], gsc) * w1);'''
new_stores = '''        const float2 achita_pair_wgs = achita_marlin_out2(d4[h], d4[2 + h], gsc);
        st_keep_f32(p.ybuf + ((size_t)(jst + tk) * HD + n0), achita_pair_wgs.x * w0);
        st_keep_f32(p.ybuf + ((size_t)(jst + tk) * HD + n0 + 8), achita_pair_wgs.y * w1);'''
assert cuda.count(old_stores) == 1
cuda = cuda.replace(old_stores, new_stores)
assert 'achita_marlin_out(' not in cuda
assert cuda.count('achita_marlin_out2(') == 12
restored = cuda.replace(new_stores, old_stores)
for paired, scalar in reversed(replacements):
    assert restored.count(paired) == 1
    restored = restored.replace(paired, scalar)
assert restored.replace(new, old) == original_cuda
literal = ast.get_source_segment(source, node.value)
assert source.count(literal) == 1
out = source.replace(literal, repr(cuda), 1)
assert out.count('okc_moe_w4a16_v216x_') == 1
out = out.replace('okc_moe_w4a16_v216x_', 'okc_moe_w4a16_v216x_pair1_')
compile(out, '<paired-moe-epilogue>', 'exec')
path = root / '.tools/moe-paired-epilogue-variant.py'
path.write_bytes(out.encode('utf-8'))
record = {'source_sha256': hashlib.sha256(out.encode('utf-8')).hexdigest(),
          'base_sha256': sha, 'paired_call_sites': 11,
          'weights_and_packing_unchanged': True,
          'normalization_intermediates': 'FP32, unchanged',
          'inverse_recovers_all_original_cuda_bytes': True,
          'rounding': 'BF16 accumulator, BF16 global scale, RN-even BF16 product, then FP32',
          'gpu_status': 'unmeasured'}
(root / '.tools/moe-paired-epilogue-source-audit.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
print(json.dumps(record))
