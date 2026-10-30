import hashlib
import json
from pathlib import Path

lines = Path('best/verified.py').read_text(encoding='utf-8').splitlines(keepends=True)
assert lines[7076].startswith("dd_decodediet_tt_mw_CONFIGS[(32,'unpooler')]")
p1 = Path('experiments/dense_p1.py').read_text(encoding='utf-8')
helpers = p1[p1.index('# --- runtime ---'):p1.index('def _p1_select')]
body = ''.join(lines[:7077]) + '\n' + helpers + '\n' + Path('experiments/dense_p4.py').read_text(encoding='utf-8')
compile(body, 'dense_p4.py', 'exec')
Path('candidates/dense_p4.py').write_bytes(body.encode('utf-8'))
sha = hashlib.sha256(body.encode()).hexdigest()
d = Path('results/dense-p4-check-1')
d.mkdir(parents=True, exist_ok=True)
(d / 'gates.json').write_text(json.dumps({
    'candidate': 'candidates/dense_p4.py', 'source_sha256': sha,
    'scope': 'confirmation of P2 near-misses only: o M8 v18s (4,8,2,1) NT1024; router M16 routertail S16/WK4',
    'pre_registered_rule': 'route iff, in this fresh run, median >= 1.015 and >= 75% of 16 paired wins, after the same correctness gates '
                           '(o: v18s_plan; router: 64 one-hot columns exact, relL2 <= 2.5e-5). Written before the check.',
    'adoption_rule': 'if both P4_USE fire and the check passes, the frozen routes become a scored-submission candidate only after a second confirming run '
                     'shows the same per-shape direction (judge per-shape ratios for (8,o) and (16,router)); expected score gain ~0.2%, inside judge CI',
    'markers': 'P4 / P4_SELECT / P4_KEEP / P4_USE'}, indent=2) + '\n', encoding='utf-8')
p = Path('DENSE_LEDGER.md')
s = p.read_text(encoding='utf-8')
row = ("| P4 | 2026-10-30 | Loop pass 4: **confirmation** (not a re-sweep) of the two P2 near-misses, each frozen to its single winning config: o M8 v18s `(4,8,2,1)` NT1024 and router M16 routertail S16/WK4. "
       "Pre-registered before launch: route iff this fresh run again shows median >= 1.015 and >= 75% wins after the same correctness gates. Expected gain \u2248 0.2% score (inside judge CI); "
       f"adoption needs a second confirming run. Candidate `candidates/dense_p4.py` sha `{sha[:8]}\u2026{sha[-4:]}` | pending | pending | | launching |\n")
p.write_text(s.rstrip('\n') + '\n' + row, encoding='utf-8')
print('sha', sha, 'bytes', len(body.encode()))
