"""Bound the value of O-only work from cached source-bound measurements."""

import json
import math

from pdl16_sources import ROOT, OUT, BASE_SHA


def main():
    result = json.loads((ROOT / 'results/decode-moe-plan-20261030/pdli-score-analysis.json').read_text())
    selected = {r['load']: result['loads'][r['load']] for r in result['case_comparisons'] if r['tokens'] in (16, 32)}
    assert len(selected) == 2
    scenarios = []
    for saving in (.5, 1., 2., 3.):
        gain = math.prod(v['ours_weighted_us'] / (v['ours_weighted_us'] - 62 * saving)
                         for v in selected.values()) ** (1 / 6)
        scenarios.append(dict(hypothetical_us_saved_in_each_of_four_layers=saving,
                              track_factor=gain, hypothetical_point=result['speedup'] * gain))
    profiles = {('decode 16', 'sliding'): 5.5, ('decode 16', 'full'): 5.8,
                ('decode 32', 'sliding'): 6.5, ('decode 32', 'full'): 5.8}
    cases = []
    for row in result['case_comparisons']:
        key = row['load'], row['layer']
        if key not in profiles:
            continue
        cases.append(dict(load=key[0], layer=key[1], layer_us=row['ours_us'],
                          profiled_O_us=profiles[key], three_percent_layer_us=.03 * row['ours_us']))
    record = dict(source_job='452fbd9df6af4ff4', source_SHA256=BASE_SHA,
        affected_loads=list(selected), cases=cases, sensitivity_scenarios=scenarios,
        decision='Do not spend the final Decode check on this O-only draft. Continue assembling the remaining handoffs or stronger component evidence.',
        scope='Hypothetical sensitivities, not measurements or a speedup forecast. Kernel profile durations are not critical-path fractions or a formal bound.',
        reason='Full-layer 3% gates require 6.4-9.4 us at full context, more than the entire recorded O duration. Real PDL can recover only exposed work; a multi-handoff candidate is needed for a credible gate pass.')
    (OUT / 'opportunity.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
