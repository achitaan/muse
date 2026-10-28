"""Check the dormant static path against pinned code before its dead-code prune.

This resolves only the listed conditional choices within that function, not a
general C preprocessor or CUDA compilation. Unknown conditions fail closed.
"""
import ast
import hashlib
import json
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'STATE.json').exists())
OUT = ROOT / 'results/moe-skinny16-selection-20261028'
raw = (OUT / 'upstream-before-pruning.py').read_bytes()
assert hashlib.sha256(raw).hexdigest() == '8bc17e47731b5dd4de9fefd3fd0db97e63e6104cacc53f01c8187c3553d03e5d'
prototype = (ROOT / 'results/moe-skinny16-preflight-20261028/prototype.py').read_bytes()
assert hashlib.sha256(prototype).hexdigest() == '921dc83ec4bdf76b0c33e777c028adfdfec84265439ca390a7369e1b6f410f96'
def cuda(source):
    return next(ast.literal_eval(n.value) for n in ast.parse(source).body if isinstance(n, ast.Assign)
                and any(isinstance(x, ast.Name) and x.id == 'CUDA_SRC' for x in n.targets))
def static_ring(source):
    key = '__device__ void skinny_ring('
    start = source.index(key, source.index(key) + 1)
    return source[start:source.index('\n}\n#endif', start) + 2]

conditions = {'defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900': True,
              'OKC_EPI4': False, 'OKC_XPF': False, '!OKC_NOLOAD': True,
              'OKC_NW == 16': False, 'OKC_NACC == 4': True, 'OKC_PRMTSC': True,
              'OKC_SKSC2': False, '!OKC_NOCOMPUTE': True, 'OKC_TWOACC': False,
              'OKC_SKT > 8': True, 'OKC_BEPI': False, 'OKC_NOLOAD': False}
def choose_branches(source):
    stack, active, output = [], True, []
    for line in source.splitlines():
        text = line.strip()
        if text.startswith('#if '):
            condition = text[4:].strip()
            assert condition in conditions, condition
            take = conditions[condition]
            stack.append([active, take]); active = active and take
        elif text.startswith('#elif '):
            condition = text[6:].strip()
            assert condition in conditions, condition
            outer, taken = stack[-1]
            active = outer and not taken and conditions[condition]
            stack[-1][1] = taken or conditions[condition]
        elif text.startswith('#else'):
            outer, taken = stack[-1]
            active = outer and not taken; stack[-1][1] = True
        elif text == '#endif':
            active = stack.pop()[0]
        elif active:
            output.append(line)
    assert not stack
    return '\n'.join(output)

expected_changes = [
    ('1.f + __bfloat162float(__ldg(p.normw', '__bfloat162float(__ldg(p.normw', 4),
    ('          const float gt = s0 * gsc, up = s1 * gsc;',
     '          const float2 achita_pair_4 = achita_marlin_out2(s0, s1, gsc);\n'
     '          const float gt = achita_pair_4.x, up = achita_pair_4.y;', 1),
    ('        const float v0f = s0 * gsc * nwc0, v1f = s1 * gsc * nwc1;',
     '        const float2 achita_pair_5 = achita_marlin_out2(s0, s1, gsc);\n'
     '        const float v0f = achita_pair_5.x * nwc0, v1f = achita_pair_5.y * nwc1;', 1),
    ('    if (two) *reinterpret_cast<float4*>(rd + 4) = make_float4(acc2[0], acc2[1], acc2[2], acc2[3]);\n', '', 1)]
for prmt in (True, False):
    conditions['OKC_PRMTSC'] = prmt
    before = choose_branches(static_ring(cuda(raw)))
    retained = choose_branches(static_ring(cuda(prototype)))
    adjusted = before
    for old, new, count in expected_changes:
        assert adjusted.count(old) == count, old
        adjusted = adjusted.replace(old, new)
    assert adjusted == retained, 'Another needed static-path operation was lost in pruning'
conditions['OKC_PRMTSC'] = True
report = dict(upstream_parent_commit='5f9f7c7b0fe83ca53f5f434e3bef2227549eaf5e',
              upstream_source_sha256=hashlib.sha256(raw).hexdigest(),
              prototype_sha256=hashlib.sha256(prototype).hexdigest(),
              listed_conditions=conditions,
              checked_prmt_modes=[True, False],
              missing_second_store_confirmed_in_unpruned_source=True,
              all_other_active_static_operations_match_after_known_precision_repairs=True,
              scope='static skinny_ring, PRMT scales, NW8/NACC4/EPI4off/SKT16 on SM90',
              limitation='Does not validate other dormant paths or CUDA execution')
(OUT / 'pruning-audit.json').write_bytes(json.dumps(report, indent=2).encode())
print(json.dumps(report))
