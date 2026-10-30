import hashlib
import json
from pathlib import Path

lines = Path('best/verified.py').read_text(encoding='utf-8').splitlines(keepends=True)
assert lines[7076].startswith("dd_decodediet_tt_mw_CONFIGS[(32,'unpooler')]")
p1 = Path('experiments/dense_p1.py').read_text(encoding='utf-8')
helpers = p1[p1.index('# --- runtime ---'):p1.index('def _p1_select')]
body = ''.join(lines[:7077]) + '\n' + helpers + '\n' + Path('experiments/dense_p2_o8.py').read_text(encoding='utf-8')
compile(body, 'dense_p2.py', 'exec')
Path('candidates/dense_p2.py').write_bytes(body.encode('utf-8'))
sha = hashlib.sha256(body.encode()).hexdigest()
d = Path('results/dense-p2-check-1')
d.mkdir(parents=True, exist_ok=True)
(d / 'gates.json').write_text(json.dumps({
    'candidate': 'candidates/dense_p2.py', 'source_sha256': sha,
    'scope': {'o m8': 'v18s c12 (4,8,2,1) NT1024 and (2,8,2,1) NT512 vs incumbent (1,8,2,1)',
              'router m8/m16': 'existing routertail_small instantiations S8/WK8 and S16/WK4 vs incumbent S16/WK8 (same module, repacked)'},
    'offline_compile': 'results/p2-offline/summary.txt: o (4,8,2,1)/(2,8,2,1) 50 regs 0 spill; (4,4,4,1) spilled -> dropped; router instantiations already in incumbent module',
    'gates': 'o: v18s_plan (bit-exact pack, one-hot, FP64). router: 64 one-hot columns == hi+lo exact, random relL2 vs FP64 <= 2.5e-5, spike row <= 2.5e-4. ABBA 16 pairs median >= 1.03, wins >= 0.75; official checker',
    'markers': 'P2 / P2_SELECT / P2_KEEP / P2_USE'}, indent=2) + '\n', encoding='utf-8')
p = Path('DENSE_LEDGER.md')
s = p.read_text(encoding='utf-8')
row = ("| P2 | 2026-10-30 | Optimizer loop pass 2 (PROGRESS.md H2 + oracle #1 small-M remainder): **o M8** v18s c12 `(4,8,2,1)` NT1024 (the o-M1 smallm2 shape) and `(2,8,2,1)` NT512 "
       "vs incumbent v18s `(1,8,2,1)` (4.5 us kernel vs 3.9 at M1); `(4,4,4,1)` dropped (spills). **router M8/M16** routertail_small S8/WK8 and S16/WK4 (already compiled in the incumbent module, repacked) vs S16/WK8. "
       "Gates: o v18s_plan; router 64 one-hot columns exact vs hi+lo, relL2 <= 2.5e-5; ABBA 16 pairs median >= 1.03, wins >= 75%. "
       f"Candidate `candidates/dense_p2.py` sha `{sha[:8]}\u2026{sha[-4:]}` | pending | pending | | launching |\n")
p.write_text(s.rstrip('\n') + '\n' + row, encoding='utf-8')
print('sha', sha, 'bytes', len(body.encode()))
