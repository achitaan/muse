from pathlib import Path

p = Path('DENSE_LEDGER.md')
s = p.read_text(encoding='utf-8')
old = 'sha `d4405270\u20261676` | pending | pending | | launching |'
assert s.count(old) == 1, s.count(old)
s = s.replace(old, 'sha `d4405270\u20261676` | check `a61fe75d5a5f427b` passed 1.5098x (runtime unchanged, all P3_KEEP). '
              '**Diagnostic (graph medians, qkv M1):** incumbent 13.3 \u00b5s; DIAG1 no-MMA 13.4 (1.002); DIAG2 no-x 13.3 (1.005); DIAG4 no-reduction 12.9 (1.038); '
              'DIAG8 weight-loads-only 12.1 (1.107); DIAG16 empty 4.9. So the kernel body is \u2248 8.4 \u00b5s, of which pure c12 weight streaming is \u2248 7.2 \u00b5s (\u2248 3.9 TB/s); '
              'the cluster reduction costs \u2248 0.4 \u00b5s; decode + MMA + x are almost fully hidden (\u2248 0.8 \u00b5s together, MMA and x individually \u2248 0). '
              '**Candidate:** ST3 0.749 / 0.747, ST4 0.914 / 0.906 at M1 / M8 (0/16 wins): SLOWED DOWN 9\u201325% | '
              '**CLOSED. Oracle #4 (cheaper c12 decode) DEAD**: decode is hidden; qkv M1 is \u2248 86% weight streaming. ST 3/4 slower |')
a = '| LOCK | pooler M1/8/16'
s = s.replace(a, '| P3 | qkv v18d more loads in flight (ST 3/4); cheaper c12 decode (oracle #4) | qkv 1/8 | `a61fe75d5a5f427b` | ST3 25% slower, ST4 9% slower; DIAG shows decode+MMA+x hidden | Do not retry ST>2 for qkv or any decode-cheapening work at M\u22648 |\n' + a, 1)
b = '| 2026-10-30 | **P2** routertail S8/WK8'
s = s.replace(b, '| 2026-10-30 | **P3** qkv v18d (1,4,8,4) ST3 / ST4 | (1, qkv_gate) / (8, qkv_gate) | v18d ST2 / qs_literal \u2248 13.3 \u00b5s graph | ST3 \u2248 17.8, ST4 \u2248 14.7 \u00b5s graph | **SLOWED DOWN 25% (ST3), 9% (ST4)** | `a61fe75d5a5f427b` |\n' + b, 1)
p.write_text(s, encoding='utf-8')
q = Path('PROGRESS.md')
t = q.read_text(encoding='utf-8')
old = '| check running | pending |'
assert t.count(old) == 1
t = t.replace(old, '| check `a61fe75d5a5f427b` passed (1.5098x, runtime unchanged) | **reverted**: ST3/ST4 9\u201325% slower. Diagnostic: qkv M1 body \u2248 8.4 \u00b5s = 7.2 weight streaming + 0.4 cluster reduction + \u2248 0.8 decode/MMA/x (hidden). Oracle #4 dead |')
q.write_text(t, encoding='utf-8')
print('ok', s.count('SLOWED DOWN'))
