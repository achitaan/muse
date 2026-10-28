"""Source-bound validation shared by the fixed Dense cluster experiment tools."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARENT = 'edc61a1e1fb26a10bcfacda3ef3fc06833c17ec7a078d5ded8eb8717509b765b'
TRIAL = '4d443b130defa8570503a1dd018e608c23a69f9e635320b81cd31ba4f82e2662'
PREFLIGHT = ROOT / 'results/dense-cluster-preflight-20261028'
TESTS = [('qkv_gate', 1), ('pooler', 7), ('o', 33), ('unpooler', 130),
         ('unpooler', 5), ('router', 8), ('router', 9), ('qkv_gate', 512), ('unpooler', 1000)]
LAYERS = ('qkv_gate', 'o', 'pooler', 'unpooler', 'router')
LOADS = (('decode 1', 1), ('decode 16', 16), ('decode 32', 32),
         ('verify 1', 8), ('verify 16', 128), ('verify 32', 256))


def read_json(path):
    return json.loads(path.read_bytes())


def source():
    raw = (ROOT / 'candidates/active.py').read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest not in (PARENT, TRIAL):
        raise RuntimeError(f'Unreviewed source {digest}; this experiment has two fixed sources')
    return raw, digest


def directory(digest):
    parent = '.scored-measurements' if digest == PARENT else '.dense-cluster-measurements'
    return ROOT / parent / digest


def full_result(result):
    """Task.md's complete named 9-test/30-benchmark suite and integrity checks."""
    body = result.get('result') or {}
    assert result.get('board') == 'muse_bf16_proj' and result.get('status') == 'done'
    assert body.get('passed') is True and body.get('gpu') == 'H200'
    assert body.get('checks', {}).get('ok') is True
    assert math.isfinite(body['speedup']) and body['speedup'] > 0
    records = body.get('records', [])
    tests = sorted((r for r in records if r.get('event') == 'test'), key=lambda r: r['index'])
    benches = sorted((r for r in records if r.get('event') == 'benchmark'), key=lambda r: r['index'])
    assert [r['index'] for r in tests] == list(range(9))
    assert [r['index'] for r in benches] == list(range(30))
    assert all(r.get('status') == 'pass' for r in tests + benches)
    assert [(r['case']['layer'], r['case']['m']) for r in tests] == TESTS
    expected = [(layer, m, load) for load, m in LOADS for layer in LAYERS]
    assert [(r['case']['layer'], r['case']['m'], r['case']['load']) for r in benches] == expected
    assert all(math.isfinite(r['speedup']) and r['speedup'] > 0 for r in benches)
    setup = [r for r in records if r.get('event') == 'setup']
    assert len(setup) == 1 and setup[0].get('status') == 'pass' and setup[0]['prepare_s'] <= 300
    return body, benches


def bound_result(digest):
    folder = directory(digest)
    if digest == PARENT and not (folder / 'result.json').exists():
        # The private repo also retains this exact published source and result.
        # A fresh checkout need not contact the service to restore its baseline.
        folder = ROOT / 'results/sprint-20261028-evidence/6131bc96427e4486'
        assert hashlib.sha256((folder / 'submission.py').read_bytes()).hexdigest() == digest
        result = read_json(folder / 'result.json')
        assert result['id'] == '6131bc96427e4486'
        return result
    assert hashlib.sha256((folder / 'submission.py').read_bytes()).hexdigest() == digest
    launch = read_json(folder / 'launch.json')
    assert launch.get('source_sha256', launch.get('local_sha256')) == digest
    result = read_json(folder / 'result.json')
    # Duplicate responses are resolved by the measurement driver, preserving the first handle.
    assert result['id'] == launch.get('resolved_job_id', launch['job_id'])
    return result
