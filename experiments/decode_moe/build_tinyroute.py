"""Freeze one small-route candidate from the retained MoE source."""

import ast
import hashlib
import json

from harness import append_harness
from tinyroute_source import derive, HERE, RESULTS, BASE_SHA


def main():
    raw, parent, modified, flags = derive()
    text = raw.decode()
    tree = ast.parse(text)
    node = next(n for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'CUDA_SRC' for t in n.targets))
    old = ast.get_source_segment(text, node)
    assert text.count(old) == 1
    text = text.replace(old, 'CUDA_SRC = ' + repr(modified))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'kernel_fn')
    launch = ast.get_source_segment(raw.decode(), fn)
    launch = launch.replace('def kernel_fn(data, state):', 'def _tr_launch(data, state):', 1)
    old = '    if TINY_DEFINES and M <= TINY_MAX_M:'
    new = ("    chosen = state.get('tr8', {}).get(M)\n"
           '    if chosen is not None:\n        ext = chosen\n'
           '    elif TINY_DEFINES and M <= TINY_MAX_M:')
    assert launch.count(old) == 1
    launch = launch.replace(old, new)
    inverse = launch.replace(new, old).replace('def _tr_launch(data, state):', 'def kernel_fn(data, state):', 1)
    assert inverse == ast.get_source_segment(raw.decode(), fn)
    text = append_harness(text) + '\n' + launch + '\n' + (HERE / 'tinyroute_runtime.py').read_text()
    ast.parse(text)
    blob = text.encode()
    assert len(blob) < 1_000_000
    (HERE / 'moe_tinyroute.py').write_bytes(blob)
    record = dict(parent_sha256=BASE_SHA, parent_CUDA_sha256=hashlib.sha256(parent.encode()).hexdigest(),
                  candidate_CUDA_sha256=hashlib.sha256(modified.encode()).hexdigest(),
                  candidate_sha256=hashlib.sha256(blob).hexdigest(), candidate_bytes=len(blob),
                  hypothesis='One-warp exact route tables remove three compute-CTA barriers at M<=8.',
                  unchanged='All GEMM, norm, accumulation, completion, weight layouts and M>8 paths.',
                  added_memory='No new timed traffic or allocation; prepare duplicates two compiled modules only.',
                  acceptance='Bitwise graph replay on 5 routing patterns x 3 changing inputs; >3% both orders on two populations.',
                  actual_path_marker='okc::moe_tr8_fused', GPU_job_launched=False,
                  stop='Reject correctness, resource, preparation-budget failure or empty selection; no parameter sweep.')
    (RESULTS / 'binding.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
