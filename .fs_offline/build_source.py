"""Reproducible assembly; arithmetic helpers copied verbatim from the parent."""
from pathlib import Path
import ast
ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'experiments/boards/moe/moe_b13.py'
def source():
    tree = ast.parse(BASE.read_bytes())
    base = ast.literal_eval(next(n.value for n in tree.body if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == 'CUDA_SRC' for t in n.targets)))
    helpers = []
    for name in ('bmul2', 'mma_bf16', 'dq4', 'achita_marlin_out2'):
        at = base.index(' '+name+'(')
        start = base.rfind('__device__', 0, at)
        helpers.append(base[start:base.index('\n}', at)+2])
    return (ROOT/'.fs_offline/fs_kernel.cu').read_text().replace(
        '// FS_HELPERS (verbatim arithmetic helpers from moe_b13)', '\n'.join(helpers))
if __name__ == '__main__':
    cu = source()
    (ROOT/'.fs_offline/fs_full.cu').write_text(cu)
    suffix = ROOT/'.fs_offline/fs_python.py'
    if suffix.exists():
        text = BASE.read_bytes().decode()+'\n\nFS_CUDA = '+repr(cu)+'\n'+suffix.read_text()
        ast.parse(text)
        assert len(text.encode()) < 1_000_000
        out = ROOT/'experiments/boards/moe/moe_fs1.py'
        out.write_bytes(text.encode())
        print(out, len(text.encode()))
