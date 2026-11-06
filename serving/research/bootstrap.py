"""Import actual archived evidence; never relabel component rows as serving."""
import hashlib
import json
from pathlib import Path
from engine import ROOT, append_record, predict

SERVING = ROOT.parent


def main():
    sources = ['results/precomp-20261105/results/s1-paired-v2.jsonl',
               'results/precomp-20261105/results/native-release-paired.jsonl',
               'results/followup-20261105/s1-isolation.jsonl',
               'results/followup-20261105/ordered-t128.jsonl']
    bindings = json.loads((SERVING/'PRECOMP_SOURCE_HASHES.json').read_text())
    rows = []
    for source in sources:
        path = SERVING/source
        file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        for line, text in enumerate(path.read_text().splitlines(), 1):
            if not text.startswith('{'):
                continue
            record = json.loads(text)
            if record.get('event') not in ('timing', 'case', 'stage', 'combined'):
                continue
            rows.append(dict(schema_version=1, id=f'{file_hash}:{line}',
                             fidelity='integrated_component' if record['event']=='timing' else 'numerical_diagnostic',
                             source=source, source_sha256=file_hash, source_line=line,
                             session='followup-20261105' if 'followup' in source else 'precomp-20261105',
                             environment_source='PRECOMP_HANDOFF.md' if 'precomp-' in source else 'results/followup-20261105/setup.log',
                             workload='synthetic; not actual serving requests',
                             serving_concurrency=None, serving_target=None,
                             values=record))
    dataset = ROOT/'historical.jsonl'
    with dataset.open('x', encoding='utf8') as f:
        for row in rows:
            f.write(json.dumps(row, allow_nan=False)+'\n')
    candidates = []
    descriptions = [
        ('s1', 'small-token finalizer fusion', 'pending_official_answer_gate',
         'Measure affected-call frequency and non-overlapped time by concurrency; then the authorized opening C16 check.'),
        ('route-workspace', 'reuse routing metadata allocation across serialized graph replays', 'unknown',
         'Inspect actual Marlin allocation lifetimes; reject if buffers are already reused or cost is outside timing.'),
        ('dflash-metadata', 'batch CPU-to-device verification metadata copies', 'unknown',
         'Inspect DFlash metadata transfer dependencies; trace exposed copies before proposing fusion.'),
        ('ordered-moe', 'ordered stream-K partial buffers', 'rejected',
         'Closed: T128 differs at GEMM1 with stable stock controls; do not repeat configuration changes.')]
    for name, mechanism, validity, experiment in descriptions:
        c = dict(schema_version=1, id=name, mechanism=mechanism, validity=validity,
                 lineage=name, by_concurrency={}, evaluation_cost=None,
                 additional_workspace_bytes=None, workspace_cap_bytes=268435456,
                 next_experiment=experiment)
        if name == 's1':
            c['patch_sha256'] = bindings['s1_postnorm_v2.patch.diff']
            c['evidence'] = ['PRECOMP_RESULTS.json', 'FOLLOWUP_RESULTS.json']
        elif name == 'ordered-moe':
            c['additional_workspace_bytes'] = 196135200
            c['evidence'] = ['FOLLOWUP_RESULTS.json']
        else:
            c['patch_sha256'] = None
            c['implementation_status'] = 'source investigation proposal; not an implemented optimization'
        candidates.append(c)
        (ROOT/(name+'.json')).write_text(json.dumps(c, indent=2)+'\n')
    (ROOT/'candidates.json').write_text(json.dumps(candidates, indent=2)+'\n')
    missing = dict(observation_rows=len(rows), serving_label_count=0,
                   serving_target_missing_fraction=1.0,
                   target_distribution=None, target_correlations=None,
                   note='No serving target distribution, residual fit, calibration or ranking accuracy can be estimated.')
    (ROOT/'data_audit.json').write_text(json.dumps(missing, indent=2)+'\n')
    append_record(ROOT/'ledger', 'prediction', predict(candidates[0]))
    print(json.dumps(missing))


if __name__ == '__main__':
    main()
