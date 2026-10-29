"""Source binding and complete named coverage for the fixed skinny16 trial."""
import ast
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARENT = '331148716e8d1ba92b45cd29725be997d44b3a5235af74d75770bb9444100d50'
TRIAL = '50dfae559fc8ce4bb6224d0dce6e7fc1ccc7a9097b7b4e19a66ea8a456efd3cd'
PROTOTYPE = '921dc83ec4bdf76b0c33e777c028adfdfec84265439ca390a7369e1b6f410f96'
PREFLIGHT = ROOT / 'results/moe-skinny16-repair-20261029'


def read_json(path):
    return json.loads(path.read_bytes())


def source():
    raw = (ROOT / 'candidates/moe_active.py').read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    assert digest in (PARENT, TRIAL), ('Unreviewed source', digest)
    return raw, digest


def directory(digest):
    return ROOT / '.moe-measurements' / digest


def verify_repair(raw, metadata=None):
    """Only two upstream #if guards may differ from the frozen failed trial."""
    meta = metadata if metadata is not None else read_json(PREFLIGHT / 'source-audit.json')
    assert hashlib.sha256(raw).hexdigest() == TRIAL == meta['source_sha256']
    node = next(n.value for n in ast.parse(raw).body if isinstance(n, ast.Assign)
                and any(isinstance(x, ast.Subscript) and ast.unparse(x) == "ALT_SRCS['sk16']" for x in n.targets))
    cuda = ast.literal_eval(node)
    edit = meta['repair']
    assert cuda.count(edit['new']) == edit['count'] == 2
    assert edit['new'] == '#if OKC_WS\n' + edit['old'] + '#endif\n'
    restored = cuda.replace(edit['new'], edit['old'])
    assert hashlib.sha256(restored.encode()).hexdigest() == edit['cuda_before_sha256']
    lines = raw.splitlines(keepends=True)
    start = sum(map(len, lines[:node.lineno - 1])) + node.col_offset
    end = sum(map(len, lines[:node.end_lineno - 1])) + node.end_col_offset
    old = raw[:start] + repr(restored).encode() + raw[end:]
    assert hashlib.sha256(old).hexdigest() == meta['repaired_from_sha256'] == 'f20caadbdafb5761c782543ad7b30351f948571ab9fc8fd5e5e8bccbb97e3f34'
    prototype = (ROOT / 'results/moe-skinny16-preflight-20261028/prototype.py').read_bytes()
    assert hashlib.sha256(prototype).hexdigest() == PROTOTYPE and old.startswith(prototype)
    # Recovering the complete old trial also protects every other CUDA/Python
    # operation and all selection/benchmark thresholds, not just the init block.
    return True


def bound_result(digest):
    folder = directory(digest)
    if digest == PARENT and not (folder / 'result.json').exists():
        folder = ROOT / 'results/resumed-moe-phase4-20261028' / digest
        assert hashlib.sha256((folder / 'submission.py').read_bytes()).hexdigest() == digest
        result = read_json(folder / 'result.json')
        assert result['id'] == 'd600231c8a8244cb'
        return result
    assert hashlib.sha256((folder / 'submission.py').read_bytes()).hexdigest() == digest
    launch = read_json(folder / 'launch.json')
    assert launch['source_sha256'] == digest
    result = read_json(folder / 'result.json')
    assert result['id'] == launch.get('resolved_job_id', launch['job_id'])
    return result


def full_result(result):
    body = result.get('result') or {}
    assert result.get('status') == 'done' and result.get('board') == 'muse_moe_w4a16'
    assert body.get('passed') is True and body.get('gpu') == 'H200' and body.get('checks', {}).get('ok') is True
    assert math.isfinite(body['speedup']) and body['speedup'] > 0
    tests = sorted((x for x in body['records'] if x.get('event') == 'test'), key=lambda x: x['index'])
    benches = sorted((x for x in body['records'] if x.get('event') == 'benchmark'), key=lambda x: x['index'])
    assert [x['index'] for x in tests] == list(range(5))
    assert [x['index'] for x in benches] == list(range(6))
    assert all(x.get('status') == 'pass' and x['case']['sharding'] == 'ep8' for x in tests + benches)
    assert [x['case']['bs'] for x in tests] == [4, 33, 257, 128, 1]
    assert tests[3]['case'].get('dist') == 'skewed'
    assert [(x['case']['bs'], x['case']['load']) for x in benches] == [
        (1, 'decode 1'), (16, 'decode 16'), (32, 'decode 32'),
        (8, 'verify 1'), (128, 'verify 16'), (256, 'verify 32')]
    assert all(math.isfinite(x['speedup']) and x['speedup'] > 0 for x in benches)
    setup = [x for x in body['records'] if x.get('event') == 'setup']
    assert len(setup) == 1 and setup[0].get('status') == 'pass' and setup[0]['prepare_s'] <= 300
    return body, benches
