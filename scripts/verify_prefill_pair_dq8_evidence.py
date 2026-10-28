"""Full Prefill validation plus fixed conservative transfer-performance guard."""
import ast
import hashlib
import json
import math
from pathlib import Path
import re
import runpy

ROOT = Path(__file__).resolve().parents[1]
runpy.run_path(str(ROOT / 'scripts/verify_prefill_evidence.py'), run_name='__main__')
PARENT = '93e57ab45fb71414c65b71a2b03241a62b3d6097890cb7612616e620c050aae7'
TRIAL = '28d04382f7224043f8cd406eda38e70e6242471673e909dbea96984fa0ea953d'
raw = (ROOT / 'candidates/prefill_active.py').read_bytes()
sha = hashlib.sha256(raw).hexdigest()
assert sha in (PARENT, TRIAL), 'This guard covers only the audited parent and exact transfer'
if sha == PARENT:
    print('Cached fully checked parent; no new GPU evaluation.')
else:
    records = []
    for digest in (PARENT, TRIAL):
        body = json.loads((ROOT / '.prefill-measurements' / digest / 'result.json').read_bytes())['result']
        records.append(body)
    old, new = records
    assert new['speedup'] > old['speedup'] * 1.015, 'Require at least 1.5% whole-track gain over cached check'
    ob = {r['index']: r for r in old['records'] if r.get('event') == 'benchmark'}
    nb = {r['index']: r for r in new['records'] if r.get('event') == 'benchmark'}
    assert set(ob) == set(nb) == set(range(8))
    ratios = {i: nb[i]['speedup'] / ob[i]['speedup'] for i in ob}
    assert min(ratios.values()) >= .97, ('Whole-layer regression', ratios)

    def profile_times(body):
        times = {}
        for r in body['records']:
            if r.get('event') != 'profile':
                continue
            lines = r.get('report', '').splitlines()
            expert_lines = [l for l in lines if 'okc_moe::moe_fused_persistent<' in l]
            assert len(expert_lines) == 1
            # Reports use a decimal microsecond column after the truncated name.
            expert = float(re.search(r'\s(\d+\.\d+)\s+\S+\s+\d+\.\d+%\s+1\s*$', expert_lines[0])[1])
            total_line = next(l for l in lines if l.startswith('total '))
            total = float(re.search(r'\s(\d+\.\d+)\s+\S+\s*$', total_line)[1])
            assert 0 < expert < total
            times[r['index']] = (expert, total - expert)
        assert set(times) == set(range(8)), 'Require executed expert-path profile in every case'
        return times

    op, np = profile_times(old), profile_times(new)
    expert_ratios = {i: op[i][0] / np[i][0] for i in op}
    control_ratios = {i: op[i][1] / np[i][1] for i in op}
    normalized = {i: expert_ratios[i] / control_ratios[i] for i in op}
    normalized_gm = math.prod(normalized.values()) ** (1 / 8)
    assert normalized_gm > 1.02, ('No material expert improvement relative to unchanged profile work', normalized_gm)
    assert min(normalized.values()) > .95, ('Expert profile regressed', normalized)
    audit = json.loads((ROOT / '.tools/prefill-pair-dq8-cpu-audit.json').read_bytes())
    assert audit['source_sha256'] == sha and audit['inverse_recovers_parent_cuda_and_other_ast']
    print(json.dumps(dict(whole_case_ratios=ratios, expert_profile_ratios=expert_ratios,
                          unchanged_profile_ratios=control_ratios, normalized_profile_ratios=normalized,
                          normalized_profile_geomean=normalized_gm, causal_significance_claim=False)))
