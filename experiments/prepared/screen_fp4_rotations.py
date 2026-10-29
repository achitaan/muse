"""Finite rotation-only screen; no claim of a general instruction bound."""
import itertools
import json
from pathlib import Path

mask = 0x81c081c0
rotations = sorted({((mask << k) | (mask >> (32-k))) & 0xffffffff for k in range(32)})
def disjoint(parts):
    return all(a & b == 0 for a, b in itertools.combinations(parts, 2))
triples = sum(disjoint(p) for p in itertools.combinations(rotations, 3))
covers = sum(disjoint(p) for p in itertools.combinations(rotations, 4))
assert len(rotations) == 16 and triples == 48 and covers == 0
report = dict(mask=hex(mask), unique_rotations=len(rotations), disjoint_triples=triples,
              four_disjoint_rotation_covers=covers, cpu_only=True,
              no_global_instruction_lower_bound=True)
(Path(__file__).resolve().parents[2] / '.tools/fp4-rotation-only-screen.json').write_bytes(
    (json.dumps(report, indent=2) + '\n').encode())
print(json.dumps(report))
