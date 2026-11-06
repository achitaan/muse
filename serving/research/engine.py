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

VERSION = 1
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
        if cost is None or not math.isfinite(cost) or cost < 0:
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


def predict(candidate):
    """Amdahl model only for an explicitly non-overlapped critical-path share.

    Replays are never training rows; unknown quantities stay null. In particular,
    reciprocal step latency is not automatically throughput under queueing.
    """
    result = dict(schema_version=VERSION, candidate=candidate['id'],
                  candidate_sha256=digest(candidate), model='conditional-amdahl-v1',
                  patch_sha256=candidate.get('patch_sha256'),
                  concurrency={}, official_score=None, aggregate_proxy=None,
                  probability_improving=None, probability_constraint_violation=None,
                  probability_status='unavailable: no calibrated serving model',
                  evaluation_cost=candidate.get('evaluation_cost'),
                  validity=candidate.get('validity', 'unknown'))
    memory = candidate.get('additional_workspace_bytes')
    cap = candidate.get('workspace_cap_bytes')
    result['workspace_feasible'] = None if memory is None or cap is None else 0 <= memory <= cap
    result['full_model_memory_feasible'] = None
    for c in CONCURRENCIES:
        features = candidate.get('by_concurrency', {}).get(str(c), {})
        required = ('critical_path_fraction', 'stage_new_old_ratio', 'added_time_fraction')
        missing = [k for k in required if features.get(k) is None]
        row = dict(latency_reduction=None, throughput_change=None, interactivity_change=None,
                   interval_kind='conditional scenario bounds; not a predictive interval', missing=missing)
        if not missing:
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
                'concurrency', 'metric', 'stock', 'candidate', 'path_executed', 'gates', 'source', 'measurement_started_at')
    if any(k not in observation for k in required):
        raise ValueError('missing observation provenance or gates')
    prior = records(directory).get(observation['prediction_id'])
    if not prior or prior['kind'] != 'prediction':
        raise ValueError('freeze a prediction before ingesting a result')
    if prior['payload']['candidate_sha256'] != observation['candidate_sha256']:
        raise ValueError('candidate changed after prediction')
    if not prior['payload'].get('patch_sha256') or prior['payload']['patch_sha256'] != observation['patch_sha256']:
        raise ValueError('observation patch does not match frozen prediction')
    started = dt.datetime.fromisoformat(observation['measurement_started_at'])
    frozen = dt.datetime.fromisoformat(prior['recorded_at'])
    if started.tzinfo is None or started <= frozen:
        raise ValueError('measurement must start after prediction freeze')
    if not isinstance(observation['path_executed'], bool):
        raise ValueError('execution evidence must be explicit')
    for row in records(directory).values():
        if row['kind'] == 'observation' and row['payload'] == observation:
            raise ValueError('duplicate observation')
    if observation['fidelity'] not in ('microbenchmark', 'integrated_component', 'full_serving', 'official_check', 'official_scored'):
        raise ValueError('unknown fidelity')
    if observation['concurrency'] not in CONCURRENCIES:
        raise ValueError('unsupported concurrency')
    if observation['metric'] not in ('latency', 'throughput', 'interactivity'):
        raise ValueError('unknown metric')
    for k in ('stock', 'candidate'):
        if isinstance(observation[k], bool) or not isinstance(observation[k], (int, float)) or not math.isfinite(observation[k]) or observation[k] <= 0:
            raise ValueError('measurements must be positive finite values')
    # Store failures too. They are evidence, never a promotion.
    return append_record(directory, 'observation', observation)


def promotion_review(observation):
    required = ('rules', 'output_agreement', 'graph_safety', 'memory', 'ttft', 'reproducible_gain')
    missing = [g for g in required if observation.get('gates', {}).get(g) is not True]
    eligible = (not missing and observation.get('path_executed') is True
                and observation.get('fidelity') in ('full_serving', 'official_scored'))
    return dict(eligible_for_human_review=eligible, missing_gates=missing,
                automatically_promoted=False)


def calibration(directory):
    data = records(directory)
    errors = []
    for row in data.values():
        if row['kind'] != 'observation':
            continue
        o = row['payload']
        if o['fidelity'] not in ('full_serving', 'official_check', 'official_scored') or o['path_executed'] is not True:
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
    return dict(prospective_comparisons=errors, trained_model=None,
                calibrated_probabilities=False,
                training_status='deferred until independent serving labels support grouped validation')


def queue(candidates):
    rows = []
    for c in candidates:
        p = predict(c)
        rejected = c.get('validity') == 'rejected' or p['workspace_feasible'] is False
        rows.append(dict(id=c['id'], status='blocked' if rejected else 'needs_evidence',
                         expected_valid_improvement_per_dollar=None,
                         next_experiment=c.get('next_experiment'), missing={k:v['missing'] for k,v in p['concurrency'].items()},
                         reason='failed correctness/feasibility' if rejected else 'no defensible calibrated utility yet'))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['predict', 'freeze', 'ingest', 'calibration', 'queue'])
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
    else:
        result = calibration(args.ledger)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
