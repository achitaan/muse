from pathlib import Path

p = Path('DENSE_LEDGER.md')
s = p.read_text(encoding='utf-8')
old = 'sha `815731e4\u20269061` | pending | pending | | launching |'
assert s.count(old) == 1, s.count(old)
s = s.replace(old, 'sha `815731e4\u20269061` | check `dc821d6f1ad94fbf` passed 1.5177x (runtime unchanged, all P2_KEEP) | '
              'o M8: (4,8,2,1) **1.025, 14/16 wins**; (2,8,2,1) 1.007 (9/16). Router M8: S8/WK8 1.012 (10/16), S16/WK4 1.004 (9/16). '
              'Router M16: S8/WK8 **0.970 (SLOWED DOWN 3%)**; S16/WK4 **1.028, 16/16 wins**. One-hot exact, relL2 2.5e-6 | '
              '**CLOSED (below 3% gate, incumbent kept).** Two consistent near-misses (\u2248 0.2 \u00b5s each, \u2248 0.1% score each) recorded in \u201ctoo small to keep\u201d |')
a = '| LOCK | pooler M1/8/16'
s = s.replace(a, '| P2 | o M8 wider v18s CTAs; router M8/M16 alternative routertail splits | o 8, router 8/16 | `dc821d6f1ad94fbf` | o 1.007\u20131.025, router M8 1.004\u20131.012, router M16 0.970 / 1.028 | All below the 3% gate; total upside \u2248 0.2% score. Do not re-sweep these neighbourhoods |\n' + a, 1)
b = '| 2026-10-30 | **P1** qkv v18d geometries'
s = s.replace(b, '| 2026-10-30 | **P2** routertail S8/WK8 | (16, router) | S16/WK8 \u2248 8.2 \u00b5s graph | \u2248 8.5 \u00b5s graph | **SLOWED DOWN 3%** (0.970, 5/16 wins) | `dc821d6f1ad94fbf` |\n' + b, 1)
c = 'Not slower but **too small to keep** (below the 3% gate): cuBLASLt neighborhoods D7/D8 (best +2.0% on (128,unpooler), +1.3% on (128,pooler)).'
assert s.count(c) == 1
s = s.replace(c, c[:-1] + '; P2 o M8 v18s (4,8,2,1) +2.5% (14/16 wins) and router M16 routertail S16/WK4 +2.8% (16/16 wins), \u2248 0.1% score each, job `dc821d6f1ad94fbf`.')
p.write_text(s, encoding='utf-8')
print('ok', s.count('SLOWED DOWN'))
