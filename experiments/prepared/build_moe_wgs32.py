"""Build a fixed 17..32-token WGS specialization from exact verified MoE."""
import ast
import hashlib
import json
from pathlib import Path

root = next(p for p in Path(__file__).resolve().parents if (p / '.git').exists())
base_sha = '331148716e8d1ba92b45cd29725be997d44b3a5235af74d75770bb9444100d50'
raw = (root / 'best/moe_verified.py').read_bytes()
assert hashlib.sha256(raw).hexdigest() == base_sha
base = raw.decode('utf-8')
tree = ast.parse(base)
literal = {n.targets[0].id: n.value.value for n in tree.body
           if isinstance(n, ast.Assign) and len(n.targets) == 1
           and isinstance(n.targets[0], ast.Name) and isinstance(n.value, ast.Constant)}
resources = '''
// Static compiler/hardware resource metadata, queried only during prepare.
std::vector<int64_t> okc_resources() {
  cudaFuncAttributes a{}, b{};
  cudaError_t ea = cudaFuncGetAttributes(&a, okc::moe_fused<false>);
  cudaError_t eb = cudaFuncGetAttributes(&b, okc::moe_fused<true>);
  TORCH_CHECK(ea == cudaSuccess && eb == cudaSuccess, "kernel resource query failed");
  return {(int64_t)a.numRegs, (int64_t)b.numRegs, (int64_t)a.localSizeBytes,
          (int64_t)b.localSizeBytes, (int64_t)a.sharedSizeBytes, (int64_t)b.sharedSizeBytes,
          (int64_t)okc_smem_bytes()};
}
'''
new_cuda = literal['CUDA_SRC'] + resources
new_cpp = literal['CPP_SRC'] + '\nstd::vector<int64_t> okc_resources();\n'
patches = []
for node in tree.body:
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        name = node.targets[0].id
        if name in ('CUDA_SRC', 'CPP_SRC'):
            patches.append((node, f'{name} = {(new_cuda if name == "CUDA_SRC" else new_cpp)!r}'))
    if not isinstance(node, ast.FunctionDef):
        continue
    if node.name == '_ext':
        for child in ast.walk(node):
            if isinstance(child, ast.Constant) and isinstance(child.value, str) and 'v216x_pair1dq8_' in child.value:
                child.value = child.value.replace('v216x_pair1dq8_', 'v216x_pair1dq8wgs32_')
            if isinstance(child, ast.keyword) and child.arg == 'functions':
                assert ast.literal_eval(child.value) == ['okc_run', 'okc_set_pdl']
                child.value.elts.append(ast.Constant('okc_resources'))
        patches.append((node, ast.unparse(node)))
    elif node.name == 'kernel_fn':
        index = next(i for i, n in enumerate(node.body) if isinstance(n, ast.If)
                     and ast.unparse(n.test).startswith('TINY_DEFINES and'))
        original_chain = node.body[index]
        prefix = ast.parse("scratch = state['scratch']\nwgs32 = state.get('wgs32')").body
        special = ast.parse("if wgs32 is not None and 17 <= M <= 32:\n    ext = wgs32['ext']\n    scratch = wgs32['scratch']").body[0]
        special.orelse = [original_chain]
        node.body[index:index + 1] = prefix + [special]
        calls = [n for n in ast.walk(node) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute) and n.func.attr == 'okc_run']
        assert len(calls) == 1 and ast.unparse(calls[0].args[1]) == "state['scratch']"
        calls[0].args[1] = ast.Name(id='scratch', ctx=ast.Load())
        patches.append((node, ast.unparse(ast.fix_missing_locations(node))))
lines = base.splitlines(keepends=True)
for node, replacement in sorted(patches, key=lambda p: p[0].lineno, reverse=True):
    lines[node.lineno - 1:node.end_lineno] = [replacement + '\n']
addon = (root / 'experiments/prepared/moe_wgs32_addon.py').read_text(encoding='utf-8')
combined = ''.join(lines) + '\n# Fixed four-tile WGS specialization; all original licenses remain above.\n' + addon
compile(combined, '<moe-wgs32>', 'exec')
out = root / '.tools/moe-wgs32-variant.py'
out.write_bytes(combined.encode('utf-8'))

# Evaluate only the pure string transformer; no module imports or GPU actions.
helper = next(n for n in ast.parse(addon).body if isinstance(n, ast.FunctionDef) and n.name == '_wgs32_cuda_source')
namespace = {}
exec(compile(ast.Module(body=[helper], type_ignores=[]), '<pure-source-transform>', 'exec'), namespace)
alternate = namespace['_wgs32_cuda_source'](new_cuda)
(root / '.tools/moe-wgs32-inspect.cu').write_bytes(alternate.encode())
assert '#define OKC_FT_T 32' in alternate and '#define OKC_FT_T 64' not in alternate
assert all(f'struct {family}<{n}>' not in alternate for family in ('WG', 'WGSMMA') for n in range(5, 9))
assert alternate.count('default: __trap(); break;') == 3
record = {'source_sha256': hashlib.sha256(out.read_bytes()).hexdigest(), 'base_sha256': base_sha,
          'packaged_bytes': out.stat().st_size, 'original_cuda_prefix_bytes_unchanged': True,
          'alternate_cuda_sha256': hashlib.sha256(alternate.encode()).hexdigest(),
          'dispatch_tokens': [17, 32], 'original_tokens_elsewhere': True,
          'max_token_tiles': [8, 4], 'x_ring_payload_bytes': [3 * 32768, 3 * 16384],
          'packed_weights': 'all original; shared with the existing variants',
          'extra_compiled_variants': 1, 'extra_scratch_banks': 1,
          'resource_fields': ['registers_small', 'registers_big', 'local_bytes_small', 'local_bytes_big',
                              'static_shared_small', 'static_shared_big', 'dynamic_shared'],
          'gpu_status': 'unmeasured'}
(root / '.tools/moe-wgs32-source-audit.json').write_bytes((json.dumps(record, indent=2) + '\n').encode())
print(json.dumps(record))
