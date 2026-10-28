"""Keep the read-only native MoE parameter object in CUDA parameter memory."""
import ast
import hashlib
import json
from pathlib import Path
import re

ROOT = next(p for p in Path(__file__).resolve().parents if (p / '.git').exists())
BASE_SHA = '331148716e8d1ba92b45cd29725be997d44b3a5235af74d75770bb9444100d50'
OLD = '__global__ void __launch_bounds__(NTL, 1) moe_fused(Params p) {'
NEW = '''// Local change: Params is immutable during the grid; keep references in
// parameter memory rather than requiring a distinct per-thread object.
__global__ void __launch_bounds__(NTL, 1) moe_fused(const __grid_constant__ Params p) {'''


def build():
    raw = (ROOT / 'best/moe_verified.py').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == BASE_SHA
    text = raw.decode('utf8')
    original_tree = ast.parse(text)
    node = next(n for n in original_tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'CUDA_SRC' for t in n.targets))
    cuda = ast.literal_eval(node.value)
    assert cuda.count(OLD) == 1 and '__grid_constant__' not in cuda
    body = cuda[cuda.index(OLD):cuda.index('\n}  // namespace okc', cuda.index(OLD))]
    assert not re.findall(r'\bp\.\w+\s*(?:[+*/-]?=(?!=)|\+\+|--)', body)
    assert not re.findall(r'(?<!const )Params&\s+p', cuda)
    assert 'const_cast' not in cuda and not re.findall(r'\bParams\s*\*', cuda)
    changed = cuda.replace(OLD, NEW, 1)
    assert changed.replace(NEW, OLD, 1) == cuda
    lines = text.splitlines(keepends=True)
    lines[node.lineno-1:node.end_lineno] = ['CUDA_SRC = ' + repr(changed) + '\n']
    result = ''.join(lines).encode('utf8')
    compile(result, '<moe-grid-params>', 'exec')
    # Independent full module recovery: every other Python/embedded source node
    # must equal the original. The complete CUDA function body stays unchanged.
    rebuilt = ast.parse(result)
    for item in rebuilt.body:
        if isinstance(item, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'CUDA_SRC' for t in item.targets):
            item.value = ast.Constant(cuda)
    assert ast.dump(rebuilt, include_attributes=False) == ast.dump(original_tree, include_attributes=False)
    assert changed[changed.index(NEW)+len(NEW):] == cuda[cuda.index(OLD)+len(OLD):]
    (ROOT / '.tools/moe-gridparams-variant.py').write_bytes(result)
    (ROOT / '.tools/moe-gridparams-inspect.cu').write_bytes(changed.encode('utf8'))
    record = {'base_sha256': BASE_SHA, 'source_sha256': hashlib.sha256(result).hexdigest(),
              'packaged_bytes': len(result), 'changed_kernel_declarations': 1,
              'kernel_body_bytes_unchanged': True, 'original_module_ast_recovered': True,
              'parameter_member_assignments': 0, 'nonconst_Refs_or_pointer_aliases_to_parameter_object': 0,
              'read_only_object_contract': 'pointer pointees remain mutable; the Params object and pointer values are immutable',
              'other_precision_weights_host_dispatch_unchanged': True,
              'gpu_status': 'unmeasured; neither parameter copying nor a speedup inferred from source alone'}
    (ROOT / '.tools/moe-gridparams-source-audit.json').write_text(json.dumps(record, indent=2)+'\n',encoding='utf8')
    print(json.dumps(record))


if __name__ == '__main__':
    build()
