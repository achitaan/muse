from pathlib import Path

p = Path('DENSE_LEDGER.md')
s = p.read_text(encoding='utf-8')
old = 'sha `240eec05\u20269e68` | pending | pending | | launching |'
assert s.count(old) == 1, s.count(old)
s = s.replace(old, 'sha `240eec05\u20269e68` | check `9d97cd89d3144181` passed 1.5092x (runtime unchanged, both P4_KEEP). '
              'o M8 (4,8,2,1) 1.010 (10/16 wins); router M16 S16/WK4 0.993 (4/16) | **NOT CONFIRMED \u2192 CLOSED.** P2 near-misses were noise; '
              'o M8 and router M16 neighbourhoods exhausted |')
a = '| LOCK | pooler M1/8/16'
s = s.replace(a, '| P4 | Confirmation of P2 near-misses (o M8 (4,8,2,1); router M16 S16/WK4) | o 8, router 16 | `9d97cd89d3144181` | 1.010 / 0.993 | Not reproducible: noise. Do not re-test |\n' + a, 1)
p.write_text(s, encoding='utf-8')
q = Path('PROGRESS.md')
t = q.read_text(encoding='utf-8')
t = t.rstrip('\n') + '\n' + ('| 4 | Confirm P2 near-misses in a fresh run (pre-registered: median >= 1.015, >= 75% wins) | `experiments/dense_p4.py` (commit `381f433`) | o v18s_plan; router one-hot exact | check `9d97cd89d3144181` passed 1.5092x | **not confirmed** (1.010 / 0.993): P2 gains were noise; closed |\n'
                             '| 5 | H5: RED 3 (reduce inside each CTA, then one DSMEM push) for qkv M1/M8; P3 showed the cluster reduction costs ~0.4 \u00b5s | `experiments/dense_p5.py` | NVRTC 3 instances 77 regs, 0 spill | queued for 00:01 UTC (today\'s 20 checks used) | pending |\n')
q.write_text(t, encoding='utf-8')
print('ok')
