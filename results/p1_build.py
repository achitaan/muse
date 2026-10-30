import hashlib
import json
from pathlib import Path

lines = Path('best/verified.py').read_text(encoding='utf-8').splitlines(keepends=True)
assert lines[7076].startswith("dd_decodediet_tt_mw_CONFIGS[(32,'unpooler')]")
body = ''.join(lines[:7077]) + '\n' + Path('experiments/dense_p1.py').read_text(encoding='utf-8')
compile(body, 'dense_p1.py', 'exec')
Path('candidates/dense_p1.py').write_bytes(body.encode('utf-8'))
sha = hashlib.sha256(body.encode()).hexdigest()
d = Path('results/dense-p1-check-1')
d.mkdir(parents=True, exist_ok=True)
(d / 'gates.json').write_text(json.dumps({
    'candidate': 'candidates/dense_p1.py', 'source_sha256': sha,
    'scope': 'qkv_gate M1/M8/M16 on v18d native c12, 7 geometries (wave balance)',
    'offline_compile': 'results/p1-offline/summary.txt: 14 instances, 63-93 regs, 0 spill',
    'gates': 'v18d_v18d_native_plan (bit-exact pack, one-hot, FP64 random); ABBA 16 pairs median >= 1.03, wins >= 0.75; official checker',
    'markers': 'P1_BUILD / P1_SELECT / P1_KEEP / P1_USE',
    'decision_rule': 'any P1_SELECT + pass -> freeze selection, confirming check; scored submission only if whole-track gain is material'},
    indent=2) + '\n', encoding='utf-8')
p = Path('DENSE_LEDGER.md')
s = p.read_text(encoding='utf-8')
old = 'markers P1_USE/P1_SELECT/P1_KEEP | pending | pending | | logged |'
assert s.count(old) == 1
s = s.replace(old, f'markers P1_USE/P1_SELECT/P1_KEEP. NVRTC 14 instances 63\u201393 regs, 0 spill. Candidate `candidates/dense_p1.py` sha `{sha[:8]}\u2026{sha[-4:]}` | pending | pending | | launching |')
p.write_text(s, encoding='utf-8')
print('sha', sha, 'bytes', len(body.encode()))
