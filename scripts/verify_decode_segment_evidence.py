"""Full decoder evidence plus complete native attention segment domain."""
import ast
from pathlib import Path
import runpy
ROOT=Path(__file__).resolve().parents[1]
runpy.run_path(str(ROOT/'scripts/verify_decode_evidence.py'),run_name='__main__')
tree=ast.parse((ROOT/'candidates/decode_active.py').read_bytes())
cuda=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_native_attn_CUDA_SRC' for t in n.targets))
assert 'int seg[F ? ((L == 1) ? MAXB : MAXSEG) : 1][3];' in cuda
assert 'S.cum[b] < ceg && n < MAXB' in cuda
assert 'mma_f8pv(' not in cuda and 'static constexpr int MINB = 2;' in cuda
fits=ast.unparse(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_native_attn__fits'))
assert "cfg['q_len'] != 8 or cfg['batch'] <= 64" in fits
assert '+ 600) < _native_attn_INDEX_LIMIT' in fits
print('Embedded attention coverage repaired; FP16 QK/PV and original two-CTA configuration retained.')
