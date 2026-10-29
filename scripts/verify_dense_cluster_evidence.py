"""Fixed Dense cluster correctness, actual-use and material changed/control guard."""
import ast
import hashlib
import json
import math
import statistics

from dense_cluster_evidence import ROOT, PARENT, PREFLIGHT, source, read_json, bound_result, full_result


def one_marker(log, marker):
    lines = [line[len(marker):].strip() for line in log.splitlines() if line.startswith(marker)]
    assert len(lines) == 1, ('Missing or ambiguous diagnostic', marker, len(lines))
    return lines[0]


def verify_oracle(layer, record):
    oracle = record['oracle']
    rows = (33, 64, 65, 127, 128) if layer == 'qkv_gate' else (65, 96, 127, 128)
    assert oracle['ok'] is True
    assert [(c[0], c[1]) for c in oracle['cases']] == [(m, p) for m in rows for p in (0, 1)]
    for m, pattern, relative, impulse, finite, worst, worst_impulse in oracle['cases']:
        assert finite is True
        assert all(math.isfinite(v) and v >= 0 for v in (relative, impulse, worst, worst_impulse))
        assert relative <= .004 and impulse <= .001 and worst <= .02 and worst_impulse <= .005
    paired = record['paired']
    assert paired['ok'] is True and not paired.get('budget')
    assert [p['pattern'] for p in paired['patterns']] == [0, 1]
    for pattern in paired['patterns']:
        old, new, samples = pattern['previous_ns'], pattern['candidate_ns'], pattern['samples']
        assert len(old) == len(new) == len(samples) == 7
        assert all(math.isfinite(v) and v > 0 for v in old + new + samples)
        computed = [n / o for n, o in zip(new, old)]
        assert all(math.isclose(x, y, rel_tol=1e-10) for x, y in zip(computed, samples))
        assert math.isclose(statistics.median(computed), pattern['ratio'], rel_tol=1e-10)
        assert pattern['ratio'] < .97, ('Synthetic paired comparison did not win', layer, pattern)


def main():
    raw, digest = source()
    result = bound_result(digest)
    body, benches = full_result(result)
    if digest == PARENT:
        print('Exact cached parent: complete 9 tests/30 benchmarks/integrity on H200; no baseline rerun.')
        return
    parent_raw = (ROOT / '.scored-measurements' / PARENT / 'submission.py').read_bytes()
    assert raw.startswith(parent_raw)
    assert hashlib.sha256((PREFLIGHT / 'submission.py').read_bytes()).hexdigest() == digest
    cpu = read_json(PREFLIGHT / 'cpu-audit.json')
    assert cpu['source_sha256'] == digest and cpu['original_parent_prefix_exact'] and cpu['cuda_pieces_pinned']
    assert cpu['old_wg_choices_proven_empty_from_published_log'] and cpu['no_weight_requantization']
    assert cpu['output_ownership_cases'] == 256 and cpu['dispatch_cases'] == 5134
    assert len(cpu['numerical_cases']) == 20
    units = read_json(PREFLIGHT / 'timing-unit-audit.json')
    assert units['source_sha256'] == digest and units['cuda_unchanged']
    assert units['restored_prepared_sha256'] == '4d443b130defa8570503a1dd018e608c23a69f9e635320b81cd31ba4f82e2662'
    log = body.get('log_tail', '')
    choices = ast.literal_eval(one_marker(log, 'DENSE_CLUSTER_CHOICES '))
    assert choices and tuple(sorted(set(choices))) == choices
    assert set(choices) <= {'qkv_gate', 'pooler'}, choices
    evidence = json.loads(one_marker(log, 'DENSE_CLUSTER_EVIDENCE '))
    assert set(choices) <= set(evidence)
    profiles = {r['index']: r.get('report', '') for r in body['records'] if r.get('event') == 'profile'}
    selected_indices = []
    for layer in choices:
        verify_oracle(layer, evidence[layer])
        index, symbol = (20, 'gqx_kernel<128, 1, 8, 3>') if layer == 'qkv_gate' else (22, 'gq_kernel<128, 1, 8, 2>')
        assert symbol in profiles.get(index, ''), ('Selected backend absent in actual profile', layer)
        selected_indices.append(index)
    baseline, old_benches = full_result(bound_result(PARENT))
    old = {r['index']: r['speedup'] for r in old_benches}
    ratios = {r['index']: r['speedup'] / old[r['index']] for r in benches}
    changed = math.prod(ratios[i] for i in selected_indices) ** (1 / len(selected_indices))
    control_indices = sorted(set(ratios) - set(selected_indices))
    controls = math.prod(ratios[i] for i in control_indices) ** (1 / len(control_indices))
    summary = dict(choices=choices, changed_geomean=changed, controls_geomean=controls,
                   normalized_geomean=changed / controls, per_case_ratios=ratios,
                   aggregate_ratio=body['speedup'] / baseline['speedup'],
                   statistical_significance_claim=False)
    print(json.dumps(summary))
    assert min(ratios[i] for i in selected_indices) >= .98, 'Selected projection regressed'
    assert changed > 1.03 and changed / controls > 1.03, 'Insufficient affected/control improvement'
    assert body['speedup'] > baseline['speedup'] * 1.015, 'Insufficient aggregate gain'
    print('Full source-bound Dense suite and fixed cluster evidence guard passed.')


if __name__ == '__main__':
    main()
