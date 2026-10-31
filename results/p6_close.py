from pathlib import Path

p = Path('DENSE_LEDGER.md')
s = p.read_text(encoding='utf-8')
old = '| launching | pending | | launching |'
assert s.count(old) == 1, s.count(old)
s = s.replace(old, '| check `a8a5e6f960064bd4` passed 1.5311x (runtime unchanged, P6_KEEP). (1,4,16,1) 0.943, (1,8,8,1) 0.912, (1,4,8,2) RED2 0.853, (1,8,4,2) RED3 0.730 (0\u20131/16 wins); (2,4,8,1) failed verification (no plan, fell back) | '
              '**CLOSED. SLOWED DOWN 6\u201327%**: the m32w WGMMA kernel stays best for unpooler M32 |')
a = '| LOCK | pooler M1/8/16'
s = s.replace(a, '| P6 | c12 GEMV (v18d MT=4) for unpooler M32 | unpooler 32 | `a8a5e6f960064bd4` | **SLOWED DOWN 6\u201327%** | Do not retry GEMV for unpooler M32 |\n' + a, 1)
b = '| 2026-10-31 | **P5** qkv v18d RED3'
s = s.replace(b, '| 2026-10-31 | **P6** v18d c12 GEMV MT=4, 4 geometries | (32, unpooler) | m32w WGMMA \u2248 20.4\u201322.0 \u00b5s graph | \u2248 22.9\u201329.0 \u00b5s graph | **SLOWED DOWN 6\u201327%** (0.943 / 0.912 / 0.853 / 0.730) | `a8a5e6f960064bd4` |\n' + b, 1)
p.write_text(s, encoding='utf-8')
q = Path('PROGRESS.md')
t = q.read_text(encoding='utf-8')
old = '| launching | pending |'
assert t.count(old) == 1
t = t.replace(old, '| check `a8a5e6f960064bd4` passed 1.5311x (runtime unchanged) | **reverted**: every GEMV layout 6\u201327% slower |')
t = t.rstrip('\n') + ('\n\n**Status after pass 6 (00:15 UTC 2026-10-31):** 0 wins in 6 passes. Incumbent-runtime checks scored 1.490\u20131.531 (noise band \u2248 \u00b11.4%). '
                      'Locally testable levers are exhausted: layout, loads in flight, reduction mode and kernel family at small M; decode is hidden; '
                      'large-M is at the cuBLAS wall. Remaining candidates are either already selected against historically (router M32 routertail) '
                      'or multi-day kernel projects. Paused for a user decision.\n')
q.write_text(t, encoding='utf-8')
print('ok', s.count('SLOWED DOWN'))
