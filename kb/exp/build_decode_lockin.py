"""Build the standalone lock-in variant without changing incumbent/build flags."""
import ast
from pathlib import Path

HERE = Path(__file__).resolve().parent
source = (HERE / 'decode_combo.py').read_text()
source = source[:source.index('def _dc_arms(key):')]
tree = ast.parse(source)
lines = source.splitlines(keepends=True)
remove = {'_kp_capture', '_kp_block', '_kp_summarize'}
for node in reversed(tree.body):
    if isinstance(node, ast.FunctionDef) and node.name in remove:
        del lines[node.lineno - 1:node.end_lineno]
source = ''.join(lines)
source = source.replace('# DEC_COMBO: independent KV/POOL candidates, native incumbent by default.',
                        '# DEC_LOCKIN: measured d1F KV / v1S POOL defaults; correctness-only fallback.')
source = source.replace('import statistics as _kp_stats\n', '')
source = source.replace('_KP_KEYS = tuple((1, L, s) for L in (1, 8) for s in (True, False))\n_DC_ROUNDS = 16',
                        "_DL_DEFAULTS = {(1, 1, False): 'KV', (1, 8, True): 'POOL'}\n_KP_KEYS = tuple(_DL_DEFAULTS)")
source += (HERE / 'decode_lockin_tail.py').read_text()
ast.parse(source)
assert len(source.encode()) < 1_000_000
out = HERE / 'decode_lockin.py'
out.write_text(source)
print(f'{out}: {out.stat().st_size} bytes; AST OK')
