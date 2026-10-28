"""Audit actual full-size WGS packs, fallback policy and exact source boundaries."""
import ast
import builtins
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import symtable
import types

import torch

root = next(p for p in Path(__file__).resolve().parents if (p / '.git').exists())
raw = (root / '.tools/moe-scale6-variant.py').read_bytes()
draft = raw.decode('utf-8')
tree = ast.parse(draft)
base = ast.parse((root / 'best/moe_verified.py').read_bytes())
record = json.loads((root / '.tools/moe-scale6-source-audit.json').read_text(encoding='utf-8'))
assert hashlib.sha256(raw).hexdigest() == record['source_sha256']
assert len(tree.body) == len(base.body) + 2
restored = copy.deepcopy(tree.body[:len(base.body)])
changes = []
for old, node in zip(base.body, restored):
    if ast.dump(old) == ast.dump(node): continue
    if isinstance(node, ast.Assign):
        assert node.targets[0].id == 'CUDA_SRC'
        cuda = node.value.value
        for patch in reversed(record['cuda_patch_ledger']):
            assert cuda.count(patch['new']) == 1
            cuda = cuda.replace(patch['new'], patch['old'], 1)
        assert cuda == old.value.value
        node.value = copy.deepcopy(old.value)
        changes.append('CUDA_SRC')
    elif node.name == '_wgs_copy':
        assert node.args.args.pop().arg == 's6' and node.args.defaults.pop().value is False
        injected = node.body.pop(-2)
        assert ast.unparse(injected) == 'if s6:\n    s = _s6_pack_scale_bytes(s)'
        changes.append('_wgs_copy')
    elif node.name == '_ext':
        for child in ast.walk(node):
            if isinstance(child, ast.Constant) and isinstance(child.value, str):
                child.value = child.value.replace('v216x_pair1dq8s6_', 'v216x_pair1dq8_')
        changes.append('_ext_name')
    elif node.name == 'prepare':
        kept = []
        for child in node.body:
            text = ast.unparse(child)
            if (text.startswith('s6 = ') or text == 'assert not s6 or prmt'
                    or text.startswith("if s6:\n    state['sfx'] +=")
                    or text == "state['scale6'] = s6" or text.startswith("print('MOE_SCALE6_WGS',")):
                continue
            for call in ast.walk(child):
                if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == '_wgs_copy':
                    assert len(call.keywords) == 1 and call.keywords[0].arg == 's6'
                    call.keywords.clear()
            kept.append(child)
        node.body = kept
        changes.append('prepare_scale_format')
    else:
        raise AssertionError(getattr(node, 'name', ast.dump(node)[:100]))
    assert ast.dump(node) == ast.dump(old)
assert ast.dump(ast.Module(body=restored, type_ignores=[])) == ast.dump(base)
assert changes == ['CUDA_SRC', '_ext_name', '_wgs_copy', 'prepare_scale_format']

functions = {'_stored_pos', '_scatter_code_bits', '_split_nibbles', '_pack_codes_nat',
             '_wgs_copy', '_s6_eligible', '_s6_pack_scale_bytes', '_scales_prmt_exact'}
namespace = {'torch': torch}
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in functions:
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<actual-pack-helper>', 'exec'), namespace)
namespace['_POS'] = namespace['_stored_pos']()
old_namespace = dict(namespace)
old_copy = next(n for n in base.body if isinstance(n, ast.FunctionDef) and n.name == '_wgs_copy')
exec(compile(ast.Module(body=[old_copy], type_ignores=[]), '<original-pack-reference>', 'exec'), old_namespace)
torch.set_num_threads(2)
generator = torch.Generator().manual_seed(20261028)
cases = []
for rows_count, order in ((8192, 'gate_up_interleaved'), (4096, 'natural'), (512, 'permuted')):
    codes = torch.randint(0, 256, (rows_count, 2048), dtype=torch.uint8, generator=generator)
    sf_bytes = torch.randint(64, 127, (rows_count, 256), dtype=torch.uint8, generator=generator)
    sf = sf_bytes.view(torch.float8_e4m3fn)
    if order == 'gate_up_interleaved':
        rb, g = torch.arange(512), torch.arange(8)
        rows = torch.stack((8 * rb[:, None] + g, 4096 + 8 * rb[:, None] + g), 1).reshape(-1)
    else:
        rows = torch.arange(rows_count) if order == 'natural' else torch.randperm(rows_count, generator=generator)
    assert namespace['_s6_eligible'](sf)
    original = old_namespace['_wgs_copy'](codes, sf, rows).reshape(rows_count // 256, 16, 4, 9216)
    packed = namespace['_wgs_copy'](codes, sf, rows, s6=True).reshape(rows_count // 256, 16, 4, 8960)
    # The known source pack reference is 33114871; codes must be identical bytes.
    assert torch.equal(packed[..., :8192], original[..., :8192])
    fallback = namespace['_wgs_copy'](codes, sf, rows, s6=False).reshape_as(original)
    assert torch.equal(fallback, original)
    p = packed[..., 8192:].reshape(rows_count // 256, 16, 4, 128, 6).long()
    bits = sum(p[..., j] << (8 * j) for j in range(6))
    scales = torch.stack([((bits >> (6 * j)) & 63) | 64 for j in range(8)], -1).to(torch.uint8).reshape(rows_count // 256, 16, 4, 1024)
    assert torch.equal(scales, original[..., 8192:])
    ordered = scales.reshape(rows_count // 256, 16, 4, 4, 4, 8, 4, 2).permute(0, 2, 4, 7, 5, 1, 3, 6).reshape(rows_count, 256)
    recovered = torch.empty_like(sf_bytes)
    recovered.index_copy_(0, rows, ordered)
    # Independent inverse permutation must recover every original FP8 byte.
    assert torch.equal(recovered, sf_bytes)
    cases.append({'rows': rows_count, 'row_order': order, 'scale_bytes_recovered': sf.numel(),
                  'nvfp4_code_bytes_unchanged': codes.numel(), 'original_pack_bytes': original.numel(),
                  'compressed_pack_bytes': packed.numel(), 'fallback_bytes_exact': True})
    del original, packed, fallback, p, bits, scales, ordered, recovered, codes, sf, sf_bytes

# The implementation's predicate, not a copy, is checked for every byte domain.
for code in range(256):
    sf = torch.tensor([code], dtype=torch.uint8).view(torch.float8_e4m3fn)
    assert namespace['_s6_eligible'](sf) == (64 <= code <= 126)

# Exercise the actual prepare control flow on CPU with spies for native packing
# and compilation. This verifies format/flag agreement, not GPU numerics.
class TorchFacade:
    cuda = types.SimpleNamespace(synchronize=lambda: None,
        get_device_properties=lambda dev: types.SimpleNamespace(multi_processor_count=132))
    def __getattr__(self, name): return getattr(torch, name)
policy = dict(namespace, torch=TorchFacade(), LOCAL_START=0, LOCAL_EXPERTS=2, HIDDEN=256,
              _make_scratch=lambda dev: [], _fat_order=lambda a, b: a, _unit_order=lambda a, b: a,
              _pack_codes=lambda *a: torch.zeros(16, dtype=torch.int32),
              _pack_scales=lambda *a: torch.zeros(16, dtype=torch.uint8))
for name in ('MAIN_DEFINES', 'MAIN_FALLBACK_DEFINES', 'TINY_DEFINES', 'TINY2_DEFINES', 'MID_DEFINES'):
    policy[name] = ast.literal_eval(next(n.value for n in tree.body if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)))
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in ('prepare', '_main_defines', '_wgs_on'):
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<prepare-policy>', 'exec'), policy)
policy_cases = []
for code, prmt_setting in ((80, True), (63, True), (0, True), (80, False)):
    compiled, copied = [], []
    policy['PRMT_SCALES'] = prmt_setting
    policy['_ext'] = lambda defines: compiled.append(tuple(defines))
    def copy_spy(codes, sf, rows, s6=False):
        copied.append(s6)
        return torch.zeros(16, dtype=torch.uint8)
    policy['_wgs_copy'] = copy_spy
    weights = {'w13': torch.zeros((2, 512, 128), dtype=torch.uint8),
               'w2': torch.zeros((2, 256, 128), dtype=torch.uint8),
               'w13_sf': torch.full((2, 512, 16), code, dtype=torch.uint8).view(torch.float8_e4m3fn),
               'w2_sf': torch.full((2, 256, 16), code, dtype=torch.uint8).view(torch.float8_e4m3fn),
               'w13_gs': torch.ones(2), 'w2_gs': torch.ones(2), 'post_norm_w': torch.zeros(256, dtype=torch.bfloat16)}
    with contextlib.redirect_stdout(io.StringIO()): state = policy['prepare'](weights)
    expected = 64 <= code <= 126 and prmt_setting
    assert state['scale6'] is expected and copied == [expected] * 4
    assert ('-DOKC_S6=1' in state['sfx']) is expected
    assert all(('-DOKC_S6=1' in d) is expected for d in compiled)
    assert len(compiled) == 4 and len(set(compiled)) == 4
    assert ('-DOKC_WGS=1' in compiled[0]) == bool(state['sfx'])
    policy_cases.append({'scale_byte': code, 'prmt_setting': prmt_setting, 's6': expected,
                         'compiled_variants': len(compiled), 'wgs_main': '-DOKC_WGS=1' in compiled[0]})

symbols = symtable.symtable(draft, 'candidate', 'exec')
known = set(symbols.get_identifiers()) | set(dir(builtins))
def bindings(table):
    for name in table.get_identifiers():
        s = table.lookup(name)
        if s.is_global() and s.is_referenced(): assert name in known, (table.get_name(), name)
    for child in table.get_children(): bindings(child)
bindings(symbols)
cuda = next(n.value.value for n in tree.body if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == 'CUDA_SRC' for t in n.targets))
assert cuda.count('constexpr int UNIT_CODES = 32768, UNIT_SC = 4096, UNIT_BYTES = UNIT_CODES + UNIT_SC;') == 1
assert cuda.count('WGS_UNIT_BYTES') == 10  # declaration, alignment assertion, seven device refs, host shared-size
assert 'WGS_PIECE = OKC_S6 ? 8960 : 9216' in cuda
assert 'const uint32_t shift = (uint32_t)(packet & 1) * 16u;' in cuda
assert '__funnelshift_r(lo, hi, shift)' in cuda and 'hi >> shift' in cuda
assert '0x60006000u' in cuda and '0x5C005C00u' in cuda
assert 'const uint32_t fields = (pair & 63u) | ((pair & 4032u) << 10);' in cuda

# The real factory must bind the same CPP declarations, distinguish format
# flags and reuse each module. The fake loader does not compile CUDA locally.
factory = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_ext')
literal = {}
for node in tree.body:
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name):
                try: literal[target.id] = ast.literal_eval(node.value)
                except (ValueError, TypeError): pass
builds = []
def loader(**kwargs):
    assert kwargs['cpp_sources'] == literal['CPP_SRC'] and kwargs['cuda_sources'] == cuda
    assert kwargs['functions'] == ['okc_run', 'okc_set_pdl']
    assert 'pair1dq8s6_' in kwargs['name'] and kwargs['name'].endswith('sm90a')
    module = types.SimpleNamespace(okc_set_pdl=lambda n: None)
    builds.append(kwargs)
    return module
factory_namespace = dict(literal, _EXTS={}, load_inline=loader,
    torch=types.SimpleNamespace(cuda=types.SimpleNamespace(get_device_capability=lambda: (9, 0))))
exec(compile(ast.Module(body=[factory], type_ignores=[]), '<real-factory>', 'exec'), factory_namespace)
variants = [tuple(policy['MAIN_DEFINES']) + ('-DOKC_PRMTSC=1', '-DOKC_S6=1'),
            tuple(policy['TINY_DEFINES']) + ('-DOKC_PRMTSC=1', '-DOKC_S6=1'),
            tuple(policy['MAIN_DEFINES']) + ('-DOKC_PRMTSC=1',), tuple(policy['MAIN_FALLBACK_DEFINES'])]
for definitions in variants:
    ext = factory_namespace['_ext'](definitions)
    assert factory_namespace['_ext'](definitions) is ext
assert len(builds) == len(variants) == len({kw['name'] for kw in builds})
result = {'source_sha256': record['source_sha256'], 'seed': 20261028, 'full_layout_cases': cases,
          'original_python_ast_recovered_after_enumerated_edits': True,
          'all_original_cuda_bytes_recovered': True, 'kernel_fn_unchanged': True,
          'eligibility_bytes_exhausted': 256, 'prepare_policy_cases': policy_cases,
          'no_extra_compiled_variants': True, 'global_bindings': True,
          'factory_flag_isolation_and_reuse': True,
          'cuda_intrinsic_reference': 'https://docs.nvidia.com/cuda/archive/12.5.0/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__INT.html',
          'gpu_status': 'unmeasured'}
(root / '.tools/moe-scale6-cpu-audit.json').write_bytes((json.dumps(result, indent=2) + '\n').encode())
print(json.dumps(result))
