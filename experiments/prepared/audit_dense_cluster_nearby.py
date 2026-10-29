"""Bind the unchanged native kernels to nearby-shape selection and test its guard."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'STATE.json').exists())
sys.path.insert(0, str(ROOT / 'scripts'))
from dense_cluster_evidence import PREFLIGHT, TRIAL
from verify_dense_cluster_evidence import verify_oracle

PRIOR = '169dbce77be377eea3c49b05428d67d3141c0a6f506f8dd23e57ad4cd7e269e3'
raw = (ROOT / '.tools/dense-cluster.py').read_bytes()
old_raw = (ROOT / 'results/dense-cluster-preflight-20261029/submission.py').read_bytes()
assert hashlib.sha256(raw).hexdigest() == TRIAL
assert hashlib.sha256(old_raw).hexdigest() == PRIOR
old, new = ast.parse(old_raw), ast.parse(raw)
old_compare = next(n for n in old.body if isinstance(n, ast.FunctionDef) and n.name == '_lg_compare')
compare = next(n for n in new.body if isinstance(n, ast.FunctionDef) and n.name == '_lg_compare')
restored = copy.deepcopy(new)
restored.body = [n for n in restored.body if not (isinstance(n, ast.Assign) and any(
    isinstance(t, ast.Name) and t.id == '_LG_COMPARE_ROWS' for t in n.targets))]
for index, node in enumerate(restored.body):
    if isinstance(node, ast.FunctionDef) and node.name == '_lg_compare':
        restored.body[index] = old_compare
assert ast.dump(restored) == ast.dump(old)
constants = lambda tree: {n.targets[0].id: ast.literal_eval(n.value) for n in tree.body
                         if isinstance(n, ast.Assign) and len(n.targets) == 1
                         and isinstance(n.targets[0], ast.Name)
                         and n.targets[0].id in ('_LG_CUDA', '_LG_CPP', '_LG_COMPARE_ROWS')}
before, after = constants(old), constants(new)
assert before['_LG_CUDA'] == after['_LG_CUDA'] and before['_LG_CPP'] == after['_LG_CPP']
events = [n for n in ast.walk(compare) if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Mult)
          and isinstance(n.left, ast.Call) and isinstance(n.left.func, ast.Attribute)
          and n.left.func.attr == 'elapsed_time']
assert len(events) == 1 and ast.literal_eval(events[0].right) == 1_000_000
evidence, rejected = {}, []
for layer, paired_rows in after['_LG_COMPARE_ROWS'].items():
    oracle_rows = (33, 64, 65, 127, 128) if layer == 'qkv_gate' else (65, 96, 127, 128)
    record = {'oracle': {'ok': True, 'cases': [
        [m, p, .001, .0001, True, .002, .0002] for m in oracle_rows for p in (0, 1)]},
        'paired': {'ok': True, 'cases': []}}
    for m in paired_rows:
        for pattern in (0, 1):
            previous = [12345.123456789013 + i for i in range(7)]
            candidate = [x * (.95 if m == 128 else .99) for x in previous]
            record['paired']['cases'].append(dict(m=m, pattern=pattern, previous_ns=previous,
                candidate_ns=candidate, ratio=statistics.median(n / o for n, o in zip(candidate, previous))))
    verify_oracle(layer, record)
    evidence[layer] = record
    mutations = []
    changed = copy.deepcopy(record); changed['paired']['cases'].pop(); mutations.append(('missing_case', changed))
    changed = copy.deepcopy(record); changed['paired']['cases'][0]['previous_ns'].pop(); mutations.append(('missing_sample', changed))
    changed = copy.deepcopy(record); changed['paired']['cases'][0]['previous_ns'][0] = float('nan'); mutations.append(('nonfinite_sample', changed))
    changed = copy.deepcopy(record); changed['paired']['cases'][0]['ratio'] = .5; mutations.append(('forged_ratio', changed))
    for m, ratio, name in [(128, .98, 'insufficient_m128_win'), (paired_rows[0], 1.021, 'nearby_regression')]:
        changed = copy.deepcopy(record)
        case = next(c for c in changed['paired']['cases'] if c['m'] == m)
        case['candidate_ns'] = [x * ratio for x in case['previous_ns']]
        case['ratio'] = statistics.median(n / o for n, o in zip(case['candidate_ns'], case['previous_ns']))
        mutations.append((name, changed))
    changed = copy.deepcopy(record); changed['oracle']['cases'][0][5] = .03; mutations.append(('numerical_row_error', changed))
    for name, changed in mutations:
        try:
            verify_oracle(layer, changed)
        except AssertionError:
            rejected.append(layer + ':' + name)
        else:
            raise AssertionError(('Guard accepted malformed evidence', layer, name))

serialized = json.dumps(evidence, separators=(',', ':'))
def padding(value):
    if isinstance(value, dict): return sum(padding(v) for v in value.values())
    if isinstance(value, list): return sum(padding(v) for v in value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return max(0, 26 - len(json.dumps(value)))
    return 0
bound = len(serialized) + padding(evidence) + len('DENSE_CLUSTER_EVIDENCE ')
assert bound < 15000, ('Diagnostic would not fit the official tail', bound)
audit = dict(source_sha256=TRIAL, prior_packet_sha256=PRIOR,
             all_nonselection_ast_unchanged=True, cuda_unchanged=True,
             compared_rows=after['_LG_COMPARE_ROWS'], paired_cases=20,
             paired_repeats=7, maximum_nearby_new_over_old=1.02,
             maximum_m128_new_over_old_exclusive=.97,
             diagnostic_bytes_bound=bound, guard_mutations_rejected=rejected,
             gpu_verified=False)
units = dict(source_sha256=TRIAL, prior_packet_sha256=PRIOR, cuda_unchanged=True,
             units='nanoseconds', cuda_event_multiplier=1_000_000)
PREFLIGHT.mkdir(parents=True, exist_ok=True)
files = {'submission.py': raw, 'source-audit.json': (ROOT / '.tools/dense-cluster-source-audit.json').read_bytes(),
         'cpu-audit.json': (ROOT / '.tools/dense-cluster-cpu-audit.json').read_bytes(),
         'kernels.cu': (ROOT / '.tools/dense-cluster.cu').read_bytes(),
         'nearby-selection-audit.json': (json.dumps(audit, indent=2) + '\n').encode(),
         'timing-unit-audit.json': (json.dumps(units, indent=2) + '\n').encode()}
for name, content in files.items():
    target = PREFLIGHT / name
    if target.exists(): assert target.read_bytes() == content, ('Frozen collision', name)
    else: target.write_bytes(content)
print(json.dumps(audit, indent=2))
