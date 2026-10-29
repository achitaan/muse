"""Freeze a prepare-only diagnostic over the exact retained MoE source."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'STATE.json').exists())
PARENT = '331148716e8d1ba92b45cd29725be997d44b3a5235af74d75770bb9444100d50'


def main():
    parent = (ROOT / 'best/moe_verified.py').read_bytes()
    assert hashlib.sha256(parent).hexdigest() == PARENT
    addon = (ROOT / 'experiments/prepared/moe_phase_diagnostic_addon.py').read_bytes()
    tree = ast.parse(addon)
    assert all(isinstance(n, (ast.Expr, ast.Import, ast.ImportFrom, ast.Assign, ast.FunctionDef)) for n in tree.body)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            assert node.name.startswith('_phase_') or node.name == 'prepare'
            assert not node.decorator_list and not node.args.defaults and not node.args.kw_defaults
        if isinstance(node, ast.Assign):
            assert len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
            assert node.targets[0].id in ('_phase_original_prepare', '_phase_original_kernel', '_PHASE_QUERY')
            assert isinstance(node.value, (ast.Constant, ast.Name))
    candidate = parent + b'\n# Prepare-only synthetic phase diagnostic; incumbent timed path retained.\n' + addon
    combined, original = ast.parse(candidate), ast.parse(parent)
    assert ast.dump(ast.Module(body=combined.body[:len(original.body)], type_ignores=[])) == ast.dump(original)
    assert [n.name for n in combined.body if isinstance(n, ast.FunctionDef)].count('kernel_fn') == 1
    cuda = next(ast.literal_eval(n.value) for n in original.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'CUDA_SRC' for t in n.targets))
    assert cuda.count('#define OKC_DEBUG 0') == 1
    assert 'mov.u64 %0, %globaltimer;' in cuda and 'DBG(5)' in cuda
    assert len(candidate) < 1_000_000
    digest = hashlib.sha256(candidate).hexdigest()
    report = dict(source_sha256=digest, parent_sha256=PARENT, bytes=len(candidate),
                  original_source_prefix_exact=True, original_prefix_module_ast_exact=True,
                  timed_kernel_fn_redefined=False, timed_path_global_bindings_changed=False,
                  diagnostic_cuda_change='Only OKC_DEBUG0->1 plus a host resource-query function in a separate extension',
                  original_timed_cuda_sha256=hashlib.sha256(cuda.encode()).hexdigest(),
                  target_shapes=[32, 128, 256], synthetic_only=True,
                  balanced_pairs_per_shape=10, cache_flush_bytes=134217728,
                  max_overhead_deviation=.05, extra_extension_builds=1,
                  purpose='Bottleneck evidence, not an optimization or scored publication',
                  gpu_verified=False)
    (ROOT / '.tools/moe-phase-diagnostic.py').write_bytes(candidate)
    (ROOT / '.tools/moe-phase-diagnostic-source-audit.json').write_bytes((json.dumps(report, indent=2) + '\n').encode())
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
