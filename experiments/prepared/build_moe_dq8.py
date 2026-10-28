"""Add the audited eight-operation exact FP4 placement to the paired candidate."""
import ast
import hashlib
import json
from pathlib import Path

root = next(p for p in Path(__file__).resolve().parents if (p / '.git').exists())
base_sha = '6862c3a54429fb4149e7fd78f63b24db71d70648a1947172826ea7f134082409'
raw = (root / 'results/resumed-moe-phase4-20261028' / base_sha / 'submission.py').read_bytes()
assert hashlib.sha256(raw).hexdigest() == base_sha
source = raw.decode('utf-8')
tree = ast.parse(source)
cuda_node = next(n for n in tree.body if isinstance(n, ast.Assign)
                 and any(isinstance(a, ast.Name) and a.id == 'CUDA_SRC' for a in n.targets))
cuda = ast.literal_eval(cuda_node.value)
start = cuda.index('// Dequantize one 32-bit word')
end = cuda.index('\n}\n', start) + 3
old_decode = cuda[start:end]
new_decode = '''// Exact FP4 placement adapted from Lee Penkman, lee101/openkernels,
// c565e373d34575c5f7df862b05f8be7619078b0f (Apache-2.0).
// The corresponding prepare-time bit permutation is a bijection of all 32 bits.
__device__ __forceinline__ void dq4(uint32_t q, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  constexpr uint32_t M = 0x81C081C0u;
  r0 = q & M;
  r1 = __funnelshift_l(q, q, 10) & M;
  r2 = __funnelshift_l(q, q, 13) & M;
  r3 = ((q & 0x001B001Bu) * 0x1084u) & M;
}
'''
cuda = cuda[:start] + new_decode + cuda[end:]
literal = ast.get_source_segment(source, cuda_node.value)
new_literal = repr(cuda)
out = source.replace(literal, new_literal, 1)
pos = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_stored_pos')
old_pos = ast.get_source_segment(source, pos)
new_pos = '''def _stored_pos():
    """Exact inverse of dq4's fixed eight-operation bit placement."""
    pos = {}
    for reg in range(4):
        for up in range(2):
            for bit, ob in enumerate((6, 7, 8, 15)):
                output_bit = ob + 16 * up
                pos[reg, up, bit] = ((output_bit - (0, 10, 13)[reg]) % 32 if reg < 3
                                     else (4, 0, 1, 3)[bit] + 16 * up)
    assert sorted(pos.values()) == list(range(32))
    return pos'''
assert out.count(old_pos) == 1
out = out.replace(old_pos, new_pos)
old_tag, new_tag = 'okc_moe_w4a16_v216x_pair1_', 'okc_moe_w4a16_v216x_pair1dq8_'
assert out.count(old_tag) == 1
out = out.replace(old_tag, new_tag)
assert out.replace(new_literal, literal).replace(new_pos, old_pos).replace(new_tag, old_tag) == source
assert cuda.replace(new_decode, old_decode) == ast.literal_eval(cuda_node.value)
notice = ('\n# Additional exact bit placement adapted from Lee Penkman (copyright 2026),\n'
          '# lee101/openkernels c565e373d34575c5f7df862b05f8be7619078b0f, Apache-2.0.\n'
          '# The complete Apache-2.0 license above applies; local integration changes packing and dq4 together.\n')
out += notice
compile(out, '<paired-dq8-moe>', 'exec')
path = root / '.tools/moe-paired-dq8-variant.py'
path.write_bytes(out.encode('utf-8'))
record = {'source_sha256': hashlib.sha256(out.encode('utf-8')).hexdigest(),
          'base_sha256': base_sha, 'changed_components': ['dq4', '_stored_pos', 'extension cache tag', 'attribution'],
          'source_inverse_exact': True, 'rounding_reductions_routing_unchanged': True,
          'weight_format': 'same number of NVFP4 bits; lossless permutation only',
          'gpu_status': 'unmeasured'}
(root / '.tools/moe-paired-dq8-source-audit.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
print(json.dumps(record))
