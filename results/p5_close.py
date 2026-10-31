from pathlib import Path

p = Path('DENSE_LEDGER.md')
s = p.read_text(encoding='utf-8')
old = '| queued for 00:01 UTC (daily checks exhausted) | pending | | queued |'
assert s.count(old) == 1, s.count(old)
s = s.replace(old, '| check `5206ecc37107426f` passed 1.5119x (runtime unchanged, all P5_KEEP). RED3 (1,4,8,4) 0.993 / 0.986, (1,8,4,4) 0.934 / 0.908, (1,8,8,2) 0.996 / 0.993 at M1 / M8 (0\u20136/16 wins) | '
              '**CLOSED. SLOWED DOWN 1\u20139%** (RED3 (1,8,4,4)); the others tie. The 0.4 \u00b5s reduction cost is not recoverable by intra-CTA pre-reduction |')
a = '| LOCK | pooler M1/8/16'
s = s.replace(a, '| P5 | qkv v18d RED3 (intra-CTA reduce then one DSMEM push) | qkv 1/8 | `5206ecc37107426f` | tie (0.986\u20130.996) to 9% slower | Do not retry reduction-mode changes for qkv |\n' + a, 1)
b = '| 2026-10-30 | **P3** qkv v18d (1,4,8,4) ST3 / ST4'
s = s.replace(b, '| 2026-10-31 | **P5** qkv v18d RED3 (1,8,4,4) | (1, qkv_gate) / (8, qkv_gate) | v18d RED2 / qs_literal \u2248 13.4 \u00b5s graph | \u2248 14.6 / 14.8 \u00b5s graph | **SLOWED DOWN 7% / 9%** (0.934 / 0.908) | `5206ecc37107426f` |\n' + b, 1)
s = s.replace('| queued after P5 | pending | | queued |', '| launching | pending | | launching |')
p.write_text(s, encoding='utf-8')
q = Path('PROGRESS.md')
t = q.read_text(encoding='utf-8')
old = "| queued for 00:01 UTC (today's 20 checks used) | pending |"
assert t.count(old) == 1
t = t.replace(old, '| check `5206ecc37107426f` passed 1.5119x (runtime unchanged) | **reverted**: RED3 ties (0.986\u20130.996) or is 7\u20139% slower |')
t = t.replace('| queued after P5 | pending |', '| launching | pending |')
q.write_text(t, encoding='utf-8')
print('ok', s.count('SLOWED DOWN'))
