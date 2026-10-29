"""CPU-only Triton 3.9.0 compile of the L2-prefetch kernel for GPUTarget(cuda, 90). No GPU launch."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'STATE.json').exists())
ADDON = ROOT / 'experiments/prepared/decode_l2pf_addon.py'
OUT = ROOT / 'results/decode-l2pf-preflight-20261029'


def kernel_source():
    text = ADDON.read_text(encoding='utf-8')
    tree = ast.parse(text)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_pfk')
    start = node.decorator_list[0].lineno - 1
    lines = text.splitlines()
    return '\n'.join(lines[start:node.end_lineno]) + '\n'


def main():
    import triton
    import triton.language as tl
    from triton.compiler import ASTSource
    from triton.backends.compiler import GPUTarget
    OUT.mkdir(parents=True, exist_ok=True)
    src = kernel_source()
    path = OUT / 'pf_kernel_src.py'
    path.write_text('import triton\nimport triton.language as tl\n' + src, encoding='utf-8')
    namespace = {}
    sys.path.insert(0, str(OUT))
    import importlib
    module = importlib.import_module('pf_kernel_src')
    fn = module._pfk
    compiled = triton.compile(ASTSource(fn, signature={'T': '*i64', 'W': 'constexpr'}, constexprs={'W': 2}),
                              target=GPUTarget('cuda', 90, 32), options={'num_warps': 2})
    ptx = compiled.asm['ptx']
    (OUT / 'pf_l2_kernel.ptx').write_text(ptx)
    (OUT / 'pf_l2_kernel.cubin').write_bytes(compiled.asm['cubin'])
    sass = ''
    tool = Path(triton.__file__).parent / 'backends/nvidia/bin/cuobjdump'
    if tool.exists():
        sass = subprocess.run([str(tool), '-sass', str(OUT / 'pf_l2_kernel.cubin')], capture_output=True, text=True).stdout
        (OUT / 'pf_l2_kernel.sass').write_text(sass)
    report = {
        'triton': triton.__version__, 'target': 'cuda/sm90/warp32', 'cpu_only': True, 'gpu_jobs': 0,
        'kernel_source_sha256': hashlib.sha256(src.encode()).hexdigest(),
        'addon_sha256': hashlib.sha256(ADDON.read_bytes().replace(b'\r\n', b'\n')).hexdigest(),
        'ptx_prefetch_count': ptx.count('prefetch.global.L2'),
        'sass_prefetch_lines': [l.strip() for l in sass.splitlines() if 'CCTL' in l or 'PREFETCH' in l.upper()][:8],
        'sass_ccctl_count': sum(1 for l in sass.splitlines() if 'CCTL' in l),
        'registers': compiled.metadata.num_regs if hasattr(compiled.metadata, 'num_regs') else None,
        'shared': compiled.metadata.shared,
        'metadata': repr(compiled.metadata),
    }
    (OUT / 'offline-compile.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in report if k != 'metadata'}, indent=2))


if __name__ == '__main__':
    main()
