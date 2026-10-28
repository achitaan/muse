"""Build one lossless six-bit WGS scale format with exact original fallback."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import re

root = next(p for p in Path(__file__).resolve().parents if (p / '.git').exists())
base_sha = '331148716e8d1ba92b45cd29725be997d44b3a5235af74d75770bb9444100d50'
raw = (root / 'best/moe_verified.py').read_bytes()
assert hashlib.sha256(raw).hexdigest() == base_sha
base = raw.decode('utf-8')
tree = ast.parse(base)
cuda_node = next(n for n in tree.body if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'CUDA_SRC' for t in n.targets))
old_cuda = cuda_node.value.value
cuda = old_cuda
ledger = []

def change(old, new):
    global cuda
    assert cuda.count(old) == 1, old[:100]
    cuda = cuda.replace(old, new, 1)
    ledger.append({'old': old, 'new': new})

change('#define OKC_DEBUG 0', '#ifndef OKC_S6\n#define OKC_S6 0\n#endif\n#define OKC_DEBUG 0')
start = cuda.index('#if OKC_WGS\n// ---------------------------------------------------------------- WGS')
end = cuda.index('\n#endif', start)
old_block = cuda[start:end]
block = re.sub(r'\bUNIT_BYTES\b', 'WGS_UNIT_BYTES', old_block)
assert block.count('WGS_UNIT_BYTES') == 7
block = block.replace('constexpr int WGS_PIECE = 9216;',
                      'constexpr int WGS_PIECE = OKC_S6 ? 8960 : 9216;\nconstexpr int WGS_UNIT_BYTES = 4 * WGS_PIECE;\nstatic_assert(!OKC_S6 || OKC_PRMTSC, "six-bit scales require the exact PRMT domain");\nstatic_assert(WGS_PIECE % 128 == 0 && WGS_UNIT_BYTES % 1024 == 0, "WGS scale layout alignment");')
helper = '''#if OKC_S6
// Each eight-scale packet is 48 bits; only aligned scalar shared loads are used.
__device__ __forceinline__ uint2 s6_scales(const unsigned char* pc, int packet) {
  const uint32_t address = smem_u32(pc + 8192 + ((3 * packet) >> 1) * 4);
  uint32_t lo, hi;
  asm volatile("ld.shared.u32 %0, [%1];" : "=r"(lo) : "r"(address));
  asm volatile("ld.shared.u32 %0, [%1];" : "=r"(hi) : "r"(address + 4));
  const uint32_t shift = (uint32_t)(packet & 1) * 16u;
  return make_uint2(__funnelshift_r(lo, hi, shift), hi >> shift);
}
#endif
'''
marker = '// one segment (stages [s0, s1) of unit u) on NTT n-tiles:'
assert block.count(marker) == 1
block = block.replace(marker, helper + marker)
old_load = 'uint2 scn = *reinterpret_cast<const uint2*>(pc + 8192 + (wp * 8 + g) * 8);'
new_load = '''uint2 scn;
#if OKC_S6
    scn = s6_scales(pc, wp * 8 + g);
#else
    scn = *reinterpret_cast<const uint2*>(pc + 8192 + (wp * 8 + g) * 8);
#endif'''
assert block.count(old_load) == 1
block = block.replace(old_load, new_load)
old_next = 'scn = *reinterpret_cast<const uint2*>(pc + 8192 + (((lg + 1) * 4 + wp) * 8 + g) * 8);'
new_next = '''#if OKC_S6
        scn = s6_scales(pc, ((lg + 1) * 4 + wp) * 8 + g);
#else
        scn = *reinterpret_cast<const uint2*>(pc + 8192 + (((lg + 1) * 4 + wp) * 8 + g) * 8);
#endif'''
assert block.count(old_next) == 1
block = block.replace(old_next, new_next)
old_scale = '''const uint32_t sw = js < 2 ? scg.x : scg.y, sb = (uint32_t)(js & 1) * 2u;
          const uint32_t sgp = __byte_perm(sw, 0u, 0x4040u | sb | ((sb + 1u) << 8)) * k16c + 0x5C005C00u;'''
new_scale = '''#if OKC_S6
          const uint32_t pair = js == 0 ? scg.x : js == 1 ? (scg.x >> 12)
                                : js == 2 ? __funnelshift_r(scg.x, scg.y, 24) : (scg.y >> 4);
          const uint32_t fields = (pair & 63u) | ((pair & 4032u) << 10);
          const uint32_t sgp = fields * k16c + 0x60006000u;
#else
          const uint32_t sw = js < 2 ? scg.x : scg.y, sb = (uint32_t)(js & 1) * 2u;
          const uint32_t sgp = __byte_perm(sw, 0u, 0x4040u | sb | ((sb + 1u) << 8)) * k16c + 0x5C005C00u;
#endif'''
assert block.count(old_scale) == 1
block = block.replace(old_scale, new_scale)
change(old_block, block)
old_smem = 'return base + (size_t)okc::WGS_NS * okc::UNIT_BYTES + (size_t)okc::WGS_NX * okc::WGS_XS;'
change(old_smem, old_smem.replace('okc::UNIT_BYTES', 'okc::WGS_UNIT_BYTES'))
# Reversing every declared CUDA edit must restore all original bytes.
restored = cuda
for patch in reversed(ledger):
    assert restored.count(patch['new']) == 1
    restored = restored.replace(patch['new'], patch['old'], 1)
assert restored == old_cuda

prototype = ast.parse((root / 'experiments/prepared/audit_moe_scale6_prototype_cpu.py').read_bytes())
eligible = copy.deepcopy(next(n for n in prototype.body if isinstance(n, ast.FunctionDef) and n.name == 'scale6_eligible'))
eligible.name = '_s6_eligible'
pack = ast.parse('''
def _s6_pack_scale_bytes(raw):
    shape = raw.shape
    values = (raw.to(torch.int64) & 63).reshape(*shape[:-1], shape[-1] // 8, 8)
    bits = sum(values[..., j] << (6 * j) for j in range(8))
    return torch.stack([(bits >> (8 * j)) & 255 for j in range(6)], dim=-1).to(torch.uint8).reshape(*shape[:-1], shape[-1] * 3 // 4).contiguous()
''').body[0]
patches = [(cuda_node, 'CUDA_SRC = ' + repr(cuda))]
for node in tree.body:
    if not isinstance(node, ast.FunctionDef):
        continue
    if node.name == '_wgs_copy':
        node.args.args.append(ast.arg(arg='s6'))
        node.args.defaults.append(ast.Constant(False))
        node.body.insert(-1, ast.parse('if s6:\n    s = _s6_pack_scale_bytes(s)').body[0])
        patches.append((node, ast.unparse(ast.fix_missing_locations(node))))
    elif node.name == 'prepare':
        anchor = next(i for i, n in enumerate(node.body)
                      if isinstance(n, ast.If) and isinstance(n.test, ast.Call)
                      and isinstance(n.test.func, ast.Name) and n.test.func.id == '_wgs_on')
        node.body.insert(anchor, ast.parse("s6 = PRMT_SCALES and _wgs_on() and _s6_eligible(weights['w13_sf'][lo:lo + cnt], weights['w2_sf'][lo:lo + cnt])").body[0])
        for child in ast.walk(node):
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Name) and child.func.id == '_wgs_copy':
                child.keywords.append(ast.keyword(arg='s6', value=ast.Name(id='s6', ctx=ast.Load())))
        sfx_i = next(i for i, n in enumerate(node.body) if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Subscript) and isinstance(t.slice, ast.Constant) and t.slice.value == 'sfx' for t in n.targets))
        node.body[sfx_i + 1:sfx_i + 1] = ast.parse('''
assert not s6 or prmt
if s6:
    state['sfx'] += ('-DOKC_S6=1',)
state['scale6'] = s6
print('MOE_SCALE6_WGS', bool(s6), flush=True)
''').body
        patches.append((node, ast.unparse(ast.fix_missing_locations(node))))
    elif node.name == '_ext':
        for child in ast.walk(node):
            if isinstance(child, ast.Constant) and isinstance(child.value, str) and 'okc_moe_w4a16_v216x_pair1dq8_' in child.value:
                child.value = child.value.replace('v216x_pair1dq8_', 'v216x_pair1dq8s6_')
        patches.append((node, ast.unparse(node)))

lines = base.splitlines(keepends=True)
for node, text in sorted(patches, key=lambda p: p[0].lineno, reverse=True):
    lines[node.lineno - 1:node.end_lineno] = [text + '\n']
combined = ''.join(lines) + '\n# Exact six-bit static FP8 scale storage; original wider-scale fallback retained.\n'
combined += ast.unparse(ast.Module(body=[eligible, pack], type_ignores=[])) + '\n'
compile(combined, '<moe-scale6>', 'exec')
out = root / '.tools/moe-scale6-variant.py'
out.write_bytes(combined.encode('utf-8'))
(root / '.tools/moe-scale6-inspect.cu').write_bytes(cuda.encode())
record = {'source_sha256': hashlib.sha256(out.read_bytes()).hexdigest(), 'base_sha256': base_sha,
          'packaged_bytes': out.stat().st_size, 'cuda_edits_reversible': True,
          'cuda_patch_ledger': ledger, 'kernel_fn_ast_unchanged': True,
          'encoding': 'eight exact low-six-bit scale fields in six bytes; restore constant 0x40',
          'eligibility': 'both local FP8 scale banks in 0x40..0x7e; otherwise unchanged original layout',
          'weight_stage_bytes': [36864, 35840], 'extra_compiled_variants': 0,
          'gpu_status': 'unmeasured'}
(root / '.tools/moe-scale6-source-audit.json').write_bytes((json.dumps(record, indent=2) + '\n').encode())
print(json.dumps({k: v for k, v in record.items() if k != 'cuda_patch_ledger'}))
