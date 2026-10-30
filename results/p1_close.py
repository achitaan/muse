from pathlib import Path

p = Path('DENSE_LEDGER.md')
s = p.read_text(encoding='utf-8')
old = 'check `e26e498c24e8487b` received 22:11Z | pending | | running |'
assert s.count(old) == 1
s = s.replace(old, 'check `e26e498c24e8487b` passed 1.4901x (runtime unchanged, all P1_KEEP; noise). All 7 built and verified. Paired base/cand: '
              'M1 0.945/0.967/0.849/0.920/0.827/0.920/0.941, M8 0.920/0.921/0.821/0.899/0.826/0.903/0.589, M16 0.744/0.650/0.696/0.859/0.739/0.732/0.353 '
              '(order (1,4,4,8),(1,8,2,8),(1,2,8,8),(1,8,4,4),(1,16,8,1),(2,4,4,8),(1,4,2,16)); 0\u20131/16 wins everywhere. '
              '**SLOWED DOWN 3\u201365%** | **CLOSED. H1 (wave quantization) REFUTED**: the incumbent (1,4,8,4) is the local optimum; qkv M1/M8 excess is not CTA balance |')
a = '| LOCK | pooler M1/8/16'
s = s.replace(a, '| P1 | qkv_gate v18d geometries with S=8/16, more warps, or WR=2 (wave balance) | qkv 1/8/16 | `e26e498c24e8487b` | **SLOWED DOWN 3\u201365%** | Incumbent (1,4,8,4) is the local optimum; do not retry qkv v18d geometry sweeps |\n' + a, 1)
b = '| 2026-10-30 | **S6** v18s RT=2'
s = s.replace(b, '| 2026-10-30 | **P1** qkv v18d geometries (1,4,4,8) (1,8,2,8) (1,2,8,8) (1,8,4,4) (1,16,8,1) (2,4,4,8) (1,4,2,16) | (1/8/16, qkv_gate) | v18d (1,4,8,4) / qs_literal / r3_qkv16 \u2248 13.3 / 13.3 / 14.1 \u00b5s graph | best 13.7 / 14.4 / 16.4 \u00b5s graph | **SLOWED DOWN: M1 3\u201317%, M8 8\u201341%, M16 14\u201365%** (0\u20131/16 wins) | `e26e498c24e8487b` |\n' + b, 1)
p.write_text(s, encoding='utf-8')
q = Path('PROGRESS.md')
t = q.read_text(encoding='utf-8')
old = '| **not run**: driver refused at 22:10Z, 4 checks left = 4 reserved | blocked (needs user OK to use reserve, or 00:00 UTC reset) |'
assert t.count(old) == 1
t = t.replace(old, '| user allowed reserved checks; check `e26e498c24e8487b` passed (1.4901x, runtime unchanged = noise) | **reverted (all KEEP): every layout slower, 3\u201365%. H1 refuted** \u2014 the incumbent geometry is the local optimum |')
q.write_text(t, encoding='utf-8')
print('ok', s.count('SLOWED DOWN'))
