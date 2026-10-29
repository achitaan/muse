"""Archive one terminal setup failure; no timing or full-suite claim."""
import hashlib
import json
from pathlib import Path

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'STATE.json').exists())
TRIAL='f382e1afd11d53edc1f203df0e58502dc704d7de15da66c4780fb0e9fd1eecaf'


def freeze(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    assert not path.exists() or path.read_bytes()==data,str(path)
    path.write_bytes(data)


def main():
    src=ROOT/'.decoder-cluster-norm-measurements'/TRIAL
    r=json.loads((src/'result.json').read_bytes());b=r['result']
    assert r['id']=='63acbd4d99324fea' and r['status']=='done' and b['passed'] is False
    setup=[v for v in b['records'] if v['event']=='setup']
    assert len(setup)==1 and setup[0]['status']=='fail'
    assert 'cluster norm numerical probe failed:1,' in setup[0]['error']
    markers=[s for s in b['log_tail'].splitlines() if s.startswith('CN_RESOURCES ')]
    assert len(markers)==1
    resources=json.loads(markers[0].split(' ',1)[1])
    assert resources==[[kind,32,0,72,248,132,1,4,256] for kind in range(4)]
    out=ROOT/'results/decoder-cluster-norm-setup-failure-20261029'
    for name in ('submission.py','launch.json','launch-intent.json','result.json','jobs-before-launch.json','quota-before-launch.json'):
        freeze(out/name,(src/name).read_bytes())
    archive=ROOT/'autoresearch-results/archive/20261029-161455'
    for p in archive.rglob('*'):
        if p.is_file():freeze(out/'controller'/p.relative_to(archive),p.read_bytes())
    for name in ('decoder_cluster_norm_evidence.py','measure_decoder_cluster_norm.py','verify_decoder_cluster_norm_evidence.py'):
        freeze(out/'guards'/name,(ROOT/'scripts'/name).read_bytes())
    protected=json.loads((ROOT/'.tools/decoder-package-protected-sha256.json').read_bytes())
    actual={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in protected}
    assert actual==protected
    freeze(out/'protected-source-restoration.json',(json.dumps(actual,indent=2)+'\n').encode())
    report=dict(source_sha256=TRIAL,job_id=r['id'],outcome='setup numerical gate failed',
        numerical_error=setup[0]['error'],native_resources=resources,native_resource_admission_passed=True,
        full_tests_run=0,full_benchmarks_run=0,completed_diagnostic_counts_unknown=True,
        speedup=None,scored_submission=False,retained_source='D9',
        trial_commit='cb78c30',revert_commit='96fd1b7',controller='df5fef03881740d88553c1ab63ecfb8a',
        archive='20261029-161455',all44protected_sources_restored=True,
        next_new_check_not_before_utc='2026-10-29T17:03:03.179453+00:00',
        decision='Diagnose source arithmetic/reduction ordering offline; do not weaken the fixed gate or retry unchanged code.')
    freeze(out/'review.json',(json.dumps(report,indent=2)+'\n').encode())
    state=json.loads((ROOT/'.tools/decoder-cluster-norm-active-session.json').read_bytes())
    state.update(active=False,terminal=True,outcome='discard',controller_archive='20261029-161455')
    (ROOT/'.tools/decoder-cluster-norm-active-session.json').write_text(json.dumps(state,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
