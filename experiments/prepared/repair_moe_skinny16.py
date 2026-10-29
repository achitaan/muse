"""Restore two pruned helper-only init guards; preserve the complete first trial."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'STATE.json').exists())
raw = (ROOT / 'results/moe-skinny16-selection-20261028/submission.py').read_bytes()
assert hashlib.sha256(raw).hexdigest() == 'f20caadbdafb5761c782543ad7b30351f948571ab9fc8fd5e5e8bccbb97e3f34'
node = next(n.value for n in ast.parse(raw).body if isinstance(n, ast.Assign)
            and any(isinstance(x, ast.Subscript) and ast.unparse(x) == "ALT_SRCS['sk16']" for x in n.targets))
cuda = ast.literal_eval(node)
old = '''      for (int k = 0; k < OKC_NS; ++k) mbar_init(S.wse + k, NW);
      for (int k = 0; k < NRB; ++k) { mbar_init(S.wsrf + k, OKC_WSB ? 1 : NW); mbar_init(S.wsre + k, 1); }
'''
new = '#if OKC_WS\n' + old + '#endif\n'
assert cuda.count(old) == 2
upstream = (ROOT / 'results/moe-skinny16-selection-20261028/upstream-before-pruning.py').read_bytes()
old_cuda = next(ast.literal_eval(n.value) for n in ast.parse(upstream).body if isinstance(n, ast.Assign)
                and any(isinstance(x, ast.Name) and x.id == 'CUDA_SRC' for x in n.targets))
assert old_cuda.count(new) == 2, 'Restore the exact upstream guards, not guessed synchronization'
fixed = cuda.replace(old, new)
lines = raw.splitlines(keepends=True)
start = sum(map(len, lines[:node.lineno - 1])) + node.col_offset
end = sum(map(len, lines[:node.end_lineno - 1])) + node.end_col_offset
output = raw[:start] + repr(fixed).encode() + raw[end:]
assert fixed.replace(new, old) == cuda
assert output[:start] + repr(fixed.replace(new, old)).encode() + output[start + len(repr(fixed).encode()):] == raw
compile(output, '<moe-skinny16-repaired>', 'exec')
(ROOT / '.tools/moe-skinny16-repaired.py').write_bytes(output)
(ROOT / '.tools/moe-skinny16-selected.py').write_bytes(output)
meta = json.loads((ROOT / 'results/moe-skinny16-selection-20261028/source-audit.json').read_bytes())
meta.update(source_sha256=hashlib.sha256(output).hexdigest(), bytes=len(output),
            repaired_from_sha256=hashlib.sha256(raw).hexdigest(),
            original_and_prototype_prefix_exact=False,
            original_parent_prefix_exact=True,
            repair=dict(old=old, new=new, count=2,
                        upstream_source_sha256=hashlib.sha256(upstream).hexdigest(),
                        cuda_before_sha256=hashlib.sha256(cuda.encode()).hexdigest(),
                        cuda_after_sha256=hashlib.sha256(fixed.encode()).hexdigest()),
            status='one diagnosed compiler repair; GPU retry pending; no arithmetic or selection changes')
(ROOT / '.tools/moe-skinny16-selection-audit.json').write_bytes(json.dumps(meta, indent=2).encode())
print(json.dumps({k: v for k, v in meta.items() if k != 'repair'}))
