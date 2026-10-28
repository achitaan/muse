"""Full Attention coverage plus fixed changed/control QK operand-swap guard."""
import hashlib
import json
import math
from pathlib import Path
from verify_attention_precision import verify

ROOT = Path(__file__).resolve().parents[1]
PARENT = '9e922c3555484081ca45a99a90284e26eccab181fb677700d8227a4b3402412f'
TRIAL = '469d8b04a043de71efe3425559692e8c8c880f36db0fa4faef3b06217141c0ad'
raw = (ROOT / 'candidates/attention_active.py').read_bytes()
digest = hashlib.sha256(raw).hexdigest()
assert digest in (PARENT, TRIAL)
verify(raw)
r = json.loads((ROOT / '.attention-measurements' / digest / 'result.json').read_bytes())
b = r.get('result') or {}
tests = [x for x in b.get('records', []) if x.get('event') == 'test']
benches = [x for x in b.get('records', []) if x.get('event') == 'benchmark']
assert r['status'] == 'done' and b.get('passed') and b.get('checks', {}).get('ok')
assert len(tests) == 16 and len(benches) == 12
assert all(x.get('status') == 'pass' for x in tests + benches)
if digest == PARENT:
    print('Cached exact parent full16/12/integrity; scored evidence reused without a baseline rerun.')
else:
    assert b['gpu'] == 'H200'
    cpu = json.loads((ROOT / '.tools/attention-qkswap-cpu-audit.json').read_bytes())
    assert cpu['source_sha256'] == digest and cpu['operand_k_coverage_exact'] and cpu['warp_transpose_bijective']
    baseline = json.loads((ROOT / '.attention-measurements' / PARENT / 'result.json').read_bytes())['result']
    old = {x['index']: x['speedup'] for x in baseline['records'] if x.get('event') == 'benchmark'}
    ratios = {x['index']: x['speedup'] / old[x['index']] for x in benches}
    assert set(ratios) == set(range(12))
    changed = math.prod(ratios[i] for i in range(6)) ** (1 / 6)
    controls = math.prod(ratios[i] for i in range(6, 12)) ** (1 / 6)
    assert min(ratios[i] for i in range(6)) >= .98, ('L1 regression', ratios)
    assert changed > 1.025 and changed / controls > 1.025, ('Insufficient changed/control improvement', changed, controls)
    assert b['speedup'] > baseline['speedup'] * 1.015, 'Insufficient aggregate improvement'
    profiles = {x['index']: x.get('report', '') for x in b['records'] if x.get('event') == 'profile'}
    assert all('attn_kernel_fp8pv<1,' in profiles.get(i, '') for i in range(6))
    print(json.dumps(dict(changed_geomean=changed, controls_geomean=controls,
                          normalized_geomean=changed / controls, per_case_ratios=ratios,
                          statistical_significance_claim=False)))
print('Complete source-bound16/12/integrity and BF16 precision guard passed.')
