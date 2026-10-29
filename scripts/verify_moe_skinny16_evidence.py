"""Fixed full-suite, actual backend, numerical and causal performance guard."""
import ast
import hashlib
import json
import math
import statistics

from moe_skinny16_evidence import ROOT, PARENT, PROTOTYPE, PREFLIGHT, source, read_json, bound_result, full_result, verify_repair


def marker(log, name):
    lines = [x[len(name):].strip() for x in log.splitlines() if x.startswith(name)]
    assert len(lines) == 1, ('Missing or ambiguous diagnostic', name, len(lines))
    return lines[0]


def verify_probes(evidence):
    assert evidence['selected'] is True and not evidence.get('skip_budget_or_device')
    assert 0 < evidence['preparation_elapsed_s'] <= 300
    probes = evidence['probes']
    assert probes['ok'] is True and not probes.get('budget')
    expected = [(m, p) for m in (9, 15, 16) for p in (0, 1, 2, 'remote')]
    assert [(x[0], x[1]) for x in probes['cases']] == expected
    for record in probes['cases']:
        if record[1] == 'remote':
            assert len(record) == 3 and record[2] is True
        else:
            m, pattern, relative, worst, finite = record
            assert finite is True and math.isfinite(relative) and math.isfinite(worst)
            assert 0 <= relative <= .005 and 0 <= worst <= .02
    paired = evidence['paired']
    assert paired['ok'] is True and not paired.get('budget')
    assert [p['pattern'] for p in paired['patterns']] == [0, 1, 2]
    ratios = []
    for p in paired['patterns']:
        old, new = p['previous_ns'], p['candidate_ns']
        assert len(old) == len(new) == 8 and all(math.isfinite(v) and v > 0 for v in old + new)
        ratio = statistics.median(n / o for n, o in zip(new, old))
        assert math.isclose(ratio, p['ratio'], rel_tol=1e-10)
        ratios.append(ratio)
    gm = math.prod(ratios) ** (1 / 3)
    assert math.isclose(gm, paired['geomean'], rel_tol=1e-10)
    assert gm < .97 and max(ratios) <= 1.02


def main():
    raw, digest = source()
    result = bound_result(digest)
    body, benches = full_result(result)
    if digest == PARENT:
        print('Exact cached parent passed full5/6/integrity on H200; no new baseline run or API call.')
        return
    prototype = (ROOT / 'results/moe-skinny16-preflight-20261028/prototype.py').read_bytes()
    assert hashlib.sha256(prototype).hexdigest() == PROTOTYPE
    assert verify_repair(raw)
    assert hashlib.sha256((PREFLIGHT / 'submission.py').read_bytes()).hexdigest() == digest
    cpu = read_json(PREFLIGHT / 'selection-cpu-audit.json')
    assert cpu['source_sha256'] == digest and cpu['original_prefix_exact']
    assert cpu['routing_cases'] == 48 and cpu['dispatch_cases'] == 770 and cpu['no_new_weight_copies']
    native = read_json(ROOT / 'results/moe-skinny16-preflight-20261028/cpu-audit.json')
    assert native['source_sha256'] == PROTOTYPE and native['missing_second_store_mutant_detected']
    assert native['second_tile_operand_words'] == 524288 and native['output_ownership_cases'] == 16
    pruning = read_json(PREFLIGHT / 'pruning-audit.json')
    assert pruning['prototype_sha256'] == PROTOTYPE and pruning['checked_prmt_modes'] == [True, False]
    assert pruning['all_other_active_static_operations_match_after_known_precision_repairs']
    helper = read_json(PREFLIGHT / 'helper-guard-cpu-audit.json')
    assert helper['source_sha256'] == digest and helper['all_helper_references_guarded']
    assert helper['references_checked'] == 11 and helper['original_compile_failure_mutant_detected']
    log = body.get('log_tail', '')
    assert ast.literal_eval(marker(log, 'MOE_SKINNY16_CHOICES ')) == (16,)
    evidence = json.loads(marker(log, 'MOE_SKINNY16_EVIDENCE '))
    verify_probes(evidence)
    profiles = {x['index']: x.get('report', '') for x in body['records'] if x.get('event') == 'profile'}
    assert 'moe_fused_sk16<false>' in profiles.get(1, ''), 'New backend absent from actual16-token profile'
    assert all('moe_fused_sk16' not in profiles.get(i, '') for i in (0, 2, 3, 4, 5))
    baseline, old_benches = full_result(bound_result(PARENT))
    old = {x['index']: x['speedup'] for x in old_benches}
    ratios = {x['index']: x['speedup'] / old[x['index']] for x in benches}
    controls = math.prod(ratios[i] for i in (0, 2, 3, 4, 5)) ** .2
    print(json.dumps(dict(affected_ratio=ratios[1], control_geomean=controls,
                          normalized_ratio=ratios[1] / controls, per_case_ratios=ratios,
                          aggregate_ratio=body['speedup'] / baseline['speedup'],
                          statistical_significance_claim=False)))
    assert ratios[1] > 1.03 and ratios[1] / controls > 1.03, 'Insufficient affected/control gain'
    assert body['speedup'] > baseline['speedup'] * 1.01, 'Insufficient whole-MoE gain'
    print('Full source-bound MoE5/6/integrity and fixed skinny16 evidence guard passed.')


if __name__ == '__main__':
    main()
