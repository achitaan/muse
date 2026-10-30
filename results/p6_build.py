import hashlib
import json
from pathlib import Path

lines = Path('best/verified.py').read_text(encoding='utf-8').splitlines(keepends=True)
assert lines[7076].startswith("dd_decodediet_tt_mw_CONFIGS[(32,'unpooler')]")
p1 = Path('experiments/dense_p1.py').read_text(encoding='utf-8')
helpers = p1[p1.index('# --- runtime ---'):p1.index('def _p1_select')]
p6 = Path('experiments/dense_p6.py').read_text(encoding='utf-8').replace('# --- runtime ---\n', '')
body = ''.join(lines[:7077]) + '\n' + helpers + '\n' + p6
compile(body, 'dense_p6.py', 'exec')
Path('candidates/dense_p6.py').write_bytes(body.encode('utf-8'))
sha = hashlib.sha256(body.encode()).hexdigest()
d = Path('results/dense-p6-check-1')
d.mkdir(parents=True, exist_ok=True)
(d / 'gates.json').write_text(json.dumps({
    'candidate': 'candidates/dense_p6.py', 'source_sha256': sha,
    'scope': 'unpooler M32 (8192x4096): v18d native c12 GEMV MT=4, geometries (1,4,16,1) (1,8,8,1) (2,4,8,1) RED0, (1,4,8,2) RED2, (1,8,4,2) RED3',
    'why': 'unpooler M32 m32w WGMMA ~17.0 us kernel vs same-size pooler M32 GEMV ~16.3 us; no GEMV ever tried for unpooler M32',
    'offline_compile': 'results/p6-offline/summary.txt: 116-128 regs, 0 spill',
    'gates': 'v18d_v18d_native_plan (bit-exact pack, one-hot, FP64 random); ABBA 16 pairs median >= 1.03, wins >= 0.75; official checker',
    'markers': 'P6 / P6_SELECT / P6_KEEP / P6_USE'}, indent=2) + '\n', encoding='utf-8')
p = Path('DENSE_LEDGER.md')
s = p.read_text(encoding='utf-8')
row = ("| P6 | 2026-10-30 | Loop pass 6: **unpooler M32** with a c12 GEMV (v18d MT=4) instead of the m32w WGMMA kernel (\u2248 17.0 \u00b5s kernel vs pooler M32 GEMV \u2248 16.3 \u00b5s on the same 8192\u00d74096-size weights). "
       "Geometries `(1,4,16,1)` `(1,8,8,1)` `(2,4,8,1)` RED0, `(1,4,8,2)` RED2, `(1,8,4,2)` RED3. Never tried (ledger has only m32w K-partition sweeps for this shape). NVRTC 116\u2013128 regs, 0 spill. "
       f"Gates: v18d plan + ABBA median >= 1.03, wins >= 75%. Candidate `candidates/dense_p6.py` sha `{sha[:8]}\u2026{sha[-4:]}` | queued after P5 | pending | | queued |\n")
p.write_text(s.rstrip('\n') + '\n' + row, encoding='utf-8')
q = Path('PROGRESS.md')
t = q.read_text(encoding='utf-8')
t = t.rstrip('\n') + '\n' + '| 6 | H6: c12 GEMV for unpooler M32 (no GEMV ever tried there) | `experiments/dense_p6.py` | NVRTC 5 instances, 0 spill | queued after P5 | pending |\n'
q.write_text(t, encoding='utf-8')
print('sha', sha, 'bytes', len(body.encode()))
