"""Build one source-bound submission without executing it or using GPU quota."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'STATE.json').exists())
PARENT = '331148716e8d1ba92b45cd29725be997d44b3a5235af74d75770bb9444100d50'
FULL = '2701a25e854caef3a46c1ae2002bd59932b2b056a05a1c0d2cf498dc5f8b6114'


def main():
    parent = (ROOT / 'best/moe_verified.py').read_bytes()
    assert hashlib.sha256(parent).hexdigest() == PARENT
    cuda = (ROOT / 'results/moe-cluster-x-device-default-20261029/full.cu').read_bytes()
    assert hashlib.sha256(cuda).hexdigest() == FULL
    addon = (ROOT / 'experiments/prepared/moe_cluster_x_addon.py').read_bytes()
    raw = parent + b'\n\n_CLUSTER_CUDA = ' + repr(cuda.decode()).encode() + b'\n' + addon
    tree = ast.parse(raw)
    parent_tree = ast.parse(parent)
    assert [ast.dump(n) for n in tree.body[:len(parent_tree.body)]] == [ast.dump(n) for n in parent_tree.body]
    assert raw.startswith(parent)
    sha = hashlib.sha256(raw).hexdigest()
    out = ROOT / 'results/moe-cluster-x-submission-preflight-20261029'
    out.mkdir(exist_ok=True)
    target = out / 'submission.py'
    assert not target.exists() or target.read_bytes() == raw, 'Frozen source collision'
    target.write_bytes(raw)
    report = dict(parent_sha256=PARENT, source_sha256=sha, source_bytes=len(raw), CUDA_sha256=FULL,
                  parent_is_exact_prefix=True, parent_AST_unchanged=True,
                  addon_sha256=hashlib.sha256(addon).hexdigest(),
                  Python_compile_only=True, kernel_execution=False, GPU_jobs=0,
                  changed_timed_paths='M>=64 and original PRMT eligibility, only after resource/probe admission',
                  unaffected_official_load_indices=[0,1,2,3], affected_official_load_indices=[4,5])
    data = (json.dumps(report, indent=2) + '\n').encode()
    audit = out / 'source-binding.json'
    assert not audit.exists() or audit.read_bytes() == data, 'Frozen binding collision'
    audit.write_bytes(data)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
