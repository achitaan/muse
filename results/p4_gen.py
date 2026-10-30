from pathlib import Path

s = Path('experiments/dense_p2_o8.py').read_text(encoding='utf-8')
head = ('\n# Dense P4: confirmation of the two P2 near-misses in a fresh process, frozen to the winning configs only:\n'
        '# o M8 v18s (4,8,2,1) NT1024 (P2: 1.025, 14/16) and router M16 routertail S16/WK4 (P2: 1.028, 16/16).\n'
        '# Pre-registered rule (written before the check): route iff this run again shows median >= 1.015 and\n'
        '# >= 75% paired wins, after the same correctness gates. Router M8 is not routed (P2 1.004).\n')
reps = [
    ('_P2_O_GEOMS = [((4, 8, 2, 1), 1024), ((2, 8, 2, 1), 512)]', '_P2_O_GEOMS = [((4, 8, 2, 1), 1024)]'),
    ('_P2_RT_ALTS = [(8, 8), (16, 4)]', '_P2_RT_ALTS = [(16, 4)]'),
    ('    for m in (8, 16):\n        x = torch.randn((m, 8192)', '    for m in (16,):\n        x = torch.randn((m, 8192)'),
]
for (a, b) in reps:
    assert s.count(a) == 1, a
    s = s.replace(a, b)
assert s.count('if med >= 1.03 and frac >= 0.75') == 2
s = s.replace('if med >= 1.03 and frac >= 0.75', 'if med >= 1.015 and frac >= 0.75')
s = s.replace("'P2", "'P4").replace('"P2', '"P4').replace("f'P2", "f'P4")
Path('experiments/dense_p4.py').write_text(head + s, encoding='utf-8')
print('ok', s.count('P4_'), s.count('P2_'))
