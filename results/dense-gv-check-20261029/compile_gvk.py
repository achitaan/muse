import sys; sys.path.insert(0, 'results/dense-gv-check-20261029')
import triton, gvk_src
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
for (wt, k, m, rows) in (('*fp32', 8192, 1, 1), ('*fp32', 8192, 8, 2), ('*bf16', 4096, 8, 4)):
    c = triton.compile(ASTSource(gvk_src._gvk, signature={'X': '*bf16', 'W': wt, 'O': '*fp32' if wt == '*fp32' else '*bf16', 'N': 'i32', 'K': 'constexpr', 'M': 'constexpr', 'ROWS': 'constexpr'}, constexprs={'K': k, 'M': m, 'ROWS': rows}), target=GPUTarget('cuda', 90, 32), options={'num_warps': 8})
    print('ok', wt, k, m, rows, 'regs', c.metadata.num_regs if hasattr(c.metadata, 'num_regs') else '?', 'shared', c.metadata.shared)
