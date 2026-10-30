import hashlib
import json
from pathlib import Path

lines = Path('best/verified.py').read_text(encoding='utf-8').splitlines(keepends=True)
assert lines[7076].startswith("dd_decodediet_tt_mw_CONFIGS[(32,'unpooler')]")
p1 = Path('experiments/dense_p1.py').read_text(encoding='utf-8')
helpers = p1[p1.index('# --- runtime ---'):p1.index('def _p1_select')]
body = ''.join(lines[:7077]) + '\n' + helpers + '\n' + Path('experiments/dense_p3.py').read_text(encoding='utf-8')
compile(body, 'dense_p3.py', 'exec')
Path('candidates/dense_p3.py').write_bytes(body.encode('utf-8'))
sha = hashlib.sha256(body.encode()).hexdigest()
d = Path('results/dense-p3-check-1')
d.mkdir(parents=True, exist_ok=True)
(d / 'gates.json').write_text(json.dumps({
    'candidate': 'candidates/dense_p3.py', 'source_sha256': sha,
    'scope': {'diagnostic': 'qkv M1 v18d (1,4,8,4) DIAG 1/2/4/8/16 vs incumbent (timing only, never routed)',
              'candidate': 'qkv M1/M8 v18d (1,4,8,4) ST 3 and 4 (incumbent ST 2)'},
    'offline_compile': 'results/p3-offline/summary.txt: 7 instances, 4-125 regs, 0 spill',
    'interpretation': 'DIAG8 (weight loads only) ~ incumbent -> memory/latency bound, decode not the cost (oracle #4 dead); '
                      'DIAG1 (no MMA) or DIAG4 (no reduction) much faster -> that stage is the excess',
    'gates': 'v18d_v18d_native_plan (bit-exact pack, one-hot, FP64 random); ABBA 16 pairs median >= 1.03, wins >= 0.75; official checker',
    'markers': 'P3_DIAG / P3 / P3_SELECT / P3_KEEP / P3_USE'}, indent=2) + '\n', encoding='utf-8')
p = Path('DENSE_LEDGER.md')
s = p.read_text(encoding='utf-8')
row = ("| P3 | 2026-10-30 | Loop pass 3 (continuous mode). (a) **Diagnostic** (oracle #4 premise): qkv M1 v18d `(1,4,8,4)` with DIAG 1 no-MMA, 2 no-x, 4 no-reduction, "
       "8 loads-only, 16 empty, timed vs incumbent, never routed. (b) **Candidate**: same geometry with ST 3/4 loads in flight (incumbent ST 2; ST was not varied in P1) at M1/M8. "
       "NVRTC 7 instances, 0 spill (ST3 110 regs, ST4 125). Gates: v18d plan + ABBA median >= 1.03, wins >= 75%. "
       f"Candidate `candidates/dense_p3.py` sha `{sha[:8]}\u2026{sha[-4:]}` | pending | pending | | launching |\n")
p.write_text(s.rstrip('\n') + '\n' + row, encoding='utf-8')
print('sha', sha, 'bytes', len(body.encode()))
