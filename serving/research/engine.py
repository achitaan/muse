"""CPU-only serving prediction ledger. No GPU or submission launcher.

Conditional ranges are scenario bounds, NOT calibrated prediction intervals.
Only prospective full-serving observations can assess serving predictions.
"""
import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path

VERSION = 2
CONCURRENCIES = (1, 4, 16, 32)
ROOT = Path(__file__).resolve().parent


def critical_path(nodes, changed=None):
    """Longest path through measured DAG; independent overlapping branches use max.

    Caller must include resource serialization edges and all launch/communication
    work. This is a fixed schedule counterfactual, not a queueing simulator.
    """
    changed = changed or {}
    if len({n['id'] for n in nodes}) != len(nodes):
        raise ValueError('duplicate DAG node')
    table = {n['id']: n for n in nodes}
    if not set(changed) <= set(table):
        raise ValueError('unknown changed node')
    done, active = {}, set()
    def visit(key):
        if key in done: return done[key]
        if key in active: raise ValueError('cyclic trace')
        if key not in table: raise ValueError('missing dependency')
        active.add(key)
        n = table[key]
        cost = changed.get(key, n['duration_us'])
        if isinstance(cost, bool) or not isinstance(cost, (int, float)) or not math.isfinite(cost) or cost < 0:
            raise ValueError('missing or invalid measured duration')
        done[key] = cost + max((visit(d) for d in n.get('dependencies', [])), default=0)
        active.remove(key)
        return done[key]
    return max((visit(k) for k in table), default=0)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def bounds(value, name, low=0, high=None):
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(name + ' must be [lower, upper]')
    a, b = value
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in value):
        raise ValueError(name + ' requires finite numbers')
    if not low <= a <= b or (high is not None and b > high):
        raise ValueError(name + ' has invalid bounds')
    return a, b


def trace_prediction(trace):
    """Fixed-DAG counterfactual with monotone endpoint bounds and path switching."""
    if trace.get('resource_edges_complete') is not True:
        raise ValueError('trace must include stream, dependency and resource serialization edges')
    if not trace.get('source'):
        raise ValueError('trace source is required')
    baseline = critical_path(trace['nodes'])
    if baseline <= 0:
        raise ValueError('positive baseline critical path required')
    changed = trace['changed_duration_us']
    endpoints = {k: bounds(v, 'changed duration') for k, v in changed.items()}
    fastest = critical_path(trace['nodes'], {k:v[0] for k,v in endpoints.items()})
    slowest = critical_path(trace['nodes'], {k:v[1] for k,v in endpoints.items()})
    if fastest <= 0:
        raise ValueError('nonpositive candidate critical path')
    return dict(baseline_us=baseline, candidate_us=[fastest, slowest],
                latency_reduction=[1-slowest/baseline, 1-fastest/baseline],
                source=trace['source'],
                scope='fixed trace schedule; does not model changed batching, queueing or topology')


def predict(candidate):
    """Amdahl model only for an explicitly non-overlapped critical-path share.

    Replays are never training rows; unknown quantities stay null. In particular,
    reciprocal step latency is not automatically throughput under queueing.
    """
    if not isinstance(candidate.get('id'), str) or not candidate['id'].strip():
        raise ValueError('candidate id required')
    by_c = candidate.get('by_concurrency', {})
    if not isinstance(by_c, dict) or not set(by_c) <= {str(c) for c in CONCURRENCIES}:
        raise ValueError('concurrency keys must be 1, 4, 16 or 32 strings')
    result = dict(schema_version=VERSION, candidate=candidate['id'],
                  candidate_sha256=digest(candidate), model='conditional-critical-path-v2',
                  patch_sha256=candidate.get('patch_sha256'),
                  evaluation_context=candidate.get('evaluation_context'),
                  concurrency={}, official_score=None, aggregate_proxy=None,
                  probability_improving=None, probability_constraint_violation=None,
                  probability_status='unavailable: no calibrated serving model',
                  evaluation_cost=candidate.get('evaluation_cost'),
                  validity=candidate.get('validity', 'unknown'))
    memory = candidate.get('additional_workspace_bytes')
    cap = candidate.get('workspace_cap_bytes')
    for name, value in (('workspace', memory), ('workspace cap', cap)):
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
            raise ValueError(name + ' must be a nonnegative integer or null')
    result['workspace_feasible'] = None if memory is None or cap is None else 0 <= memory <= cap
    result['full_model_memory_feasible'] = None
    for c in CONCURRENCIES:
        features = by_c.get(str(c), {})
        if not isinstance(features, dict):
            raise ValueError('per-concurrency features must be an object')
        required = ('critical_path_fraction', 'stage_new_old_ratio', 'added_time_fraction')
        missing = [k for k in required if features.get(k) is None]
        row = dict(latency_reduction=None, throughput_change=None, interactivity_change=None,
                   interval_kind='conditional scenario bounds; not a predictive interval', missing=missing)
        if features.get('trace') is not None:
            if any(features.get(k) is not None for k in required):
                raise ValueError('choose trace or serial Amdahl inputs, not both')
            traced = trace_prediction(features['trace'])
            row.update(latency_reduction=traced['latency_reduction'], missing=[], trace=traced)
            if features.get('throughput_assumption') == 'fixed_work_no_queueing_or_batch_change':
                lo, hi = traced['candidate_us']
                row['throughput_change'] = [traced['baseline_us']/hi-1, traced['baseline_us']/lo-1]
        elif not missing:
            if features.get('overlap_basis') != 'nonoverlapped_critical_path':
                raise ValueError('sum of kernel durations is not a critical-path fraction')
            f = bounds(features['critical_path_fraction'], 'fraction', high=1)
            r = bounds(features['stage_new_old_ratio'], 'ratio')
            a = bounds(features['added_time_fraction'], 'added time')
            ratios = [1 - x + x * y + z for x in f for y in r for z in a]
            if min(ratios) <= 0:
                raise ValueError('nonpositive total latency')
            row['latency_reduction'] = [1 - max(ratios), 1 - min(ratios)]
            if features.get('throughput_assumption') == 'fixed_work_no_queueing_or_batch_change':
                row['throughput_change'] = [1 / max(ratios) - 1, 1 / min(ratios) - 1]
            row['assumptions'] = features
        result['concurrency'][str(c)] = row
    return result


def append_record(directory, kind, payload):
    """Exclusive creation: never replace an earlier prediction or observation."""
    directory.mkdir(parents=True, exist_ok=True)
    body = dict(schema_version=VERSION, kind=kind,
                recorded_at=dt.datetime.now(dt.timezone.utc).isoformat(), payload=payload)
    key = digest(body)
    with (directory / (key + '.json')).open('x', encoding='utf8') as f:
        json.dump(body, f, indent=2, allow_nan=False)
        f.write('\n')
    return key


def records(directory):
    result = {}
    for p in sorted(directory.glob('*.json')):
        body = json.loads(p.read_text())
        if digest(body) != p.stem:
            raise ValueError('ledger tampering: ' + str(p))
        result[p.stem] = body
    return result


def ingest(directory, observation):
    required = ('prediction_id', 'candidate_sha256', 'patch_sha256', 'sglang_revision',
                'environment', 'gpu', 'workload', 'session', 'lineage', 'fidelity',
                'concurrency', 'metric', 'unit', 'stock', 'candidate', 'path_executed', 'gates', 'source',
                'measurement_started_at', 'measurement_id', 'baseline_patch_sha256')
    if any(k not in observation for k in required):
        raise ValueError('missing observation provenance or gates')
    prior = records(directory).get(observation['prediction_id'])
    if not prior or prior['kind'] != 'prediction':
        raise ValueError('freeze a prediction before ingesting a result')
    if prior['payload']['candidate_sha256'] != observation['candidate_sha256']:
        raise ValueError('candidate changed after prediction')
    if not prior['payload'].get('patch_sha256') or prior['payload']['patch_sha256'] != observation['patch_sha256']:
        raise ValueError('observation patch does not match frozen prediction')
    context = prior['payload'].get('evaluation_context')
    if not isinstance(context, dict):
        raise ValueError('freeze an evaluation context before measuring; legacy records remain historical')
    for key in ('sglang_revision', 'environment', 'gpu', 'workload', 'baseline_patch_sha256'):
        if not context.get(key) or context[key] != observation[key]:
            raise ValueError('evaluation context mismatch: ' + key)
    for key in ('source', 'session', 'lineage', 'measurement_id'):
        if not isinstance(observation[key], str) or not observation[key].strip():
            raise ValueError('nonempty provenance required: ' + key)
    if not isinstance(observation['gates'], dict) or any(type(v) is not bool and v is not None for v in observation['gates'].values()):
        raise ValueError('gates must contain boolean or null evidence')
    started = dt.datetime.fromisoformat(observation['measurement_started_at'])
    frozen = dt.datetime.fromisoformat(prior['recorded_at'])
    if started.tzinfo is None or started <= frozen:
        raise ValueError('measurement must start after prediction freeze')
    if started > dt.datetime.now(dt.timezone.utc):
        raise ValueError('measurement start is in the future')
    if not isinstance(observation['path_executed'], bool):
        raise ValueError('execution evidence must be explicit')
    for row in records(directory).values():
        if row['kind'] == 'observation' and row['payload'].get('measurement_id') == observation['measurement_id']:
            raise ValueError('duplicate observation')
    if observation['fidelity'] not in ('microbenchmark', 'integrated_component', 'full_serving', 'official_check', 'official_scored'):
        raise ValueError('unknown fidelity')
    if type(observation['concurrency']) is not int or observation['concurrency'] not in CONCURRENCIES:
        raise ValueError('unsupported concurrency')
    if observation['metric'] not in ('latency', 'throughput', 'interactivity'):
        raise ValueError('unknown metric')
    if observation['unit'] != {'latency':'us', 'throughput':'tokens/s', 'interactivity':'tokens/s_at_stock_interactivity'}[observation['metric']]:
        raise ValueError('metric/unit mismatch')
    for k in ('stock', 'candidate'):
        if isinstance(observation[k], bool) or not isinstance(observation[k], (int, float)) or not math.isfinite(observation[k]) or observation[k] <= 0:
            raise ValueError('measurements must be positive finite values')
    # Store failures too. They are evidence, never a promotion.
    return append_record(directory, 'observation', observation)


def promotion_review(observation):
    required = ('rules', 'output_agreement', 'graph_safety', 'memory', 'ttft', 'reproducible_gain')
    missing = [g for g in required if observation.get('gates', {}).get(g) is not True]
    try:
        stock, candidate = observation['stock'], observation['candidate']
        numeric = all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in (stock, candidate))
        gain = numeric and (candidate < stock if observation.get('metric') == 'latency'
                            else candidate > stock if observation.get('metric') in ('throughput', 'interactivity') else False)
    except KeyError:
        gain = False
    eligible = (not missing and gain and observation.get('path_executed') is True
                and observation.get('fidelity') in ('full_serving', 'official_scored'))
    return dict(eligible_for_human_review=eligible, missing_gates=missing,
                observed_improvement=bool(gain), automatically_promoted=False)


def grouped_components(rows):
    """Lineage OR session links, including transitive links, must not cross folds.

    Leakage audit: a shared session can link otherwise distinct candidate lineages;
    grouping by the pair (lineage, session) would leak information across folds.
    """
    parents = list(range(len(rows)))
    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    seen = {}
    for i, row in enumerate(rows):
        for field in ('lineage', 'session'):
            if not isinstance(row.get(field), str) or not row[field]:
                raise ValueError('missing grouping key: ' + field)
            key = field, row[field]
            if key in seen:
                parents[find(i)] = find(seen[key])
            seen[key] = i
    groups = {}
    for i in range(len(rows)):
        groups.setdefault(find(i), []).append(i)
    return sorted(groups.values(), key=lambda g:g[0])


def validation_plan(rows):
    groups = grouped_components(rows)
    return dict(independent_components=len(groups), groups=groups,
                status='insufficient_independent_groups' if len(groups)<2 else 'split_ready_not_statistically_validated',
                folds=[] if len(groups)<2 else [dict(test=g, train=[i for h in groups if h != g for i in h]) for g in groups])


def calibration(directory):
    data = records(directory)
    errors = []
    excluded = []
    for row in data.values():
        if row['kind'] != 'observation':
            continue
        o = row['payload']
        if o['fidelity'] not in ('full_serving', 'official_check', 'official_scored') or o['path_executed'] is not True:
            continue
        if any(o.get('gates', {}).get(k) is not True for k in ('rules', 'output_agreement')):
            excluded.append(dict(measurement_id=o.get('measurement_id'), reason='rules or output agreement not established'))
            continue
        p = data[o['prediction_id']]['payload']['concurrency'][str(o['concurrency'])]
        key = {'latency': 'latency_reduction', 'throughput': 'throughput_change', 'interactivity': 'interactivity_change'}[o['metric']]
        interval = p[key]
        if interval is None:
            continue
        actual = 1-o['candidate']/o['stock'] if o['metric']=='latency' else o['candidate']/o['stock']-1
        errors.append(dict(lineage=o['lineage'], session=o['session'], concurrency=o['concurrency'],
                           metric=o['metric'], actual=actual, midpoint_error=(sum(interval)/2)-actual,
                           inside_scenario_bounds=interval[0] <= actual <= interval[1]))
    return dict(prospective_comparisons=errors, excluded=excluded, grouped_validation=validation_plan(errors), trained_model=None,
                calibrated_probabilities=False,
                training_status='deferred until independent serving labels support grouped validation')


def queue(candidates):
    rows = []
    for c in candidates:
        p = predict(c)
        rejected = c.get('validity') == 'rejected' or p['workspace_feasible'] is False
        parked = c.get('research_status') == 'parked'
        rows.append(dict(id=c['id'], status='blocked' if rejected else 'parked' if parked else 'needs_evidence',
                         expected_valid_improvement_per_dollar=None,
                         next_experiment=c.get('next_experiment'), missing={k:v['missing'] for k,v in p['concurrency'].items()},
                         reason='failed correctness/feasibility' if rejected else c.get('decision_reason') if parked else 'no defensible calibrated utility yet'))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['predict', 'freeze', 'ingest', 'calibration', 'queue', 'trace', 'split', 'review'])
    p.add_argument('--input', type=Path)
    p.add_argument('--ledger', type=Path, default=ROOT/'ledger')
    args = p.parse_args()
    value = json.loads(args.input.read_text()) if args.input else None
    if args.command in ('predict', 'freeze'):
        result = predict(value)
        if args.command == 'freeze':
            result = dict(prediction_id=append_record(args.ledger, 'prediction', result), prediction=result)
    elif args.command == 'ingest':
        result = dict(observation_id=ingest(args.ledger, value))
    elif args.command == 'queue':
        result = queue(value)
    elif args.command == 'trace':
        result = trace_prediction(value)
    elif args.command == 'split':
        result = validation_plan(value)
    elif args.command == 'review':
        result = promotion_review(value)
    else:
        result = calibration(args.ledger)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
