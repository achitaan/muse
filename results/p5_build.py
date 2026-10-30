import hashlib
import json
from pathlib import Path

lines = Path('best/verified.py').read_text(encoding='utf-8').splitlines(keepends=True)
assert lines[7076].startswith("dd_decodediet_tt_mw_CONFIGS[(32,'unpooler')]")
p1 = Path('experiments/dense_p1.py').read_text(encoding='utf-8')
helpers = p1[p1.index('# --- runtime ---'):p1.index('def _p1_select')]
p5 = Path('experiments/dense_p5.py').read_text(encoding='utf-8').replace('# --- runtime ---\n', '')
body = ''.join(lines[:7077]) + '\n' + helpers + '\n' + p5
compile(body, 'dense_p5.py', 'exec')
Path('candidates/dense_p5.py').write_bytes(body.encode('utf-8'))
sha = hashlib.sha256(body.encode()).hexdigest()
d = Path('results/dense-p5-check-1')
d.mkdir(parents=True, exist_ok=True)
(d / 'gates.json').write_text(json.dumps({
    'candidate': 'candidates/dense_p5.py', 'source_sha256': sha,
    'scope': 'qkv M1/M8 v18d RED 3 (intra-CTA reduce, then one DSMEM push per row group) for (1,4,8,4), (1,8,4,4), (1,8,8,2)',
    'why': 'P3 DIAG4 (no reduction) 1.038 on qkv M1: cluster reduction ~0.4 us; RED 3 cuts DSMEM traffic by WK x',
    'offline_compile': 'results/p5-offline-red3/summary.txt: 77 regs, 0 spill',
    'gates': 'v18d_v18d_native_plan (bit-exact pack, one-hot, FP64 random); ABBA 16 pairs median >= 1.03, wins >= 0.75; official checker',
    'markers': 'P5 / P5_SELECT / P5_KEEP / P5_USE'}, indent=2) + '\n', encoding='utf-8')
p = Path('DENSE_LEDGER.md')
s = p.read_text(encoding='utf-8')
row = ("| P5 | 2026-10-30 | Loop pass 5: qkv M1/M8 v18d **RED 3** (intra-CTA reduce, then one DSMEM push per row group, instead of RED 2's per-warp pushes) for `(1,4,8,4)` (incumbent geometry), `(1,8,4,4)`, `(1,8,8,2)`. "
       "Motivation: P3 DIAG4 showed the cluster reduction costs \u2248 0.4 \u00b5s (1.038). RED 3 never tested for qkv (S1 tested RED3 only on the sliced unpooler). NVRTC 77 regs, 0 spill. "
       f"Gates: v18d plan + ABBA median >= 1.03, wins >= 75%. Candidate `candidates/dense_p5.py` sha `{sha[:8]}\u2026{sha[-4:]}` | queued for 00:01 UTC (daily checks exhausted) | pending | | queued |\n")
p.write_text(s.rstrip('\n') + '\n' + row, encoding='utf-8')
print('sha', sha, 'bytes', len(body.encode()))
