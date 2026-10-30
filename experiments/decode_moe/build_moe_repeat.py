"""Freeze one diagnostic job; it cannot select or publish the rejected TR8."""

import ast
import argparse
import base64
import hashlib
import json
import zlib

from harness import append_harness
from tinyroute_source import derive, HERE, BASE_SHA

OUT = HERE.parents[1] / 'results/decode-moe-plan-20261030/moe-repeat'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--compact', action='store_true')
    args = parser.parse_args()
    out = OUT.parent / 'moe-repeat-v2' if args.compact else OUT
    raw, parent_cuda, tr8_cuda, flags = derive()
    assert hashlib.sha256(raw).hexdigest() == BASE_SHA
    source = raw.decode()
    tree = ast.parse(source)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_ext')
    old = ast.get_source_segment(source, fn)
    ext = old
    edits = [('def _ext(', 'def _mr_variant_ext('), ('_EXTS', '_MR_VARIANT_EXTS'),
             ('src = CUDA_SRC', 'src = _MR_TR_SRC'),
             ('okc_moe_w4a16_v216x_pair1dq8_', 'okc_mr_tr8_observer_')]
    for a, b in edits:
        assert a in ext
        ext = ext.replace(a, b)
    inverse = ext
    for a, b in reversed(edits): inverse = inverse.replace(b, a)
    assert inverse == old
    failed = (HERE / 'moe_tinyroute.py').read_text()
    failed_tree = ast.parse(failed)
    launch = next(n for n in failed_tree.body if isinstance(n, ast.FunctionDef) and n.name == '_tr_launch')
    launch_src = ast.get_source_segment(failed, launch).replace('def _tr_launch(', 'def _mr_variant_launch(', 1)
    rt = (HERE / 'tinyroute_runtime.py').read_text()
    rt_tree = ast.parse(rt)
    fixtures = '\n\n'.join(ast.get_source_segment(rt,n) for n in rt_tree.body
        if isinstance(n, ast.FunctionDef) and n.name in ['_tr_probe_data','_tr_mutate'])
    text = source + append_harness('') + '\n_MR_VARIANT_EXTS = {}\n'
    text += '_MR_TR_SRC = __import__("zlib").decompress(__import__("base64").b64decode(' + repr(base64.b64encode(zlib.compress(tr8_cuda.encode(),9)).decode()) + ')).decode()\n'
    text += ext + '\n' + launch_src + '\n' + fixtures + '\n'
    text += (HERE / ('moe_repeat_v2_runtime.py' if args.compact else 'moe_repeat_runtime.py')).read_text()
    blob = text.encode(); ast.parse(blob)
    assert blob.startswith(raw) and len(blob) < 1_000_000
    # The official timed kernel_fn is never redefined.
    assert len([n for n in ast.parse(blob).body if isinstance(n, ast.FunctionDef) and n.name == 'kernel_fn']) == 1
    out.mkdir(exist_ok=True)
    (HERE / ('moe_repeat_diag_v2.py' if args.compact else 'moe_repeat_diag.py')).write_bytes(blob)
    record = dict(source_sha256=hashlib.sha256(blob).hexdigest(), bytes=len(blob),
        parent_sha256=BASE_SHA, full_parent_prefix_identical=True, timed_kernel_fn_unchanged=True,
        parent_CUDA_sha256=hashlib.sha256(parent_cuda.encode()).hexdigest(),
        observed_TR8_CUDA_sha256=hashlib.sha256(tr8_cuda.encode()).hexdigest(),
        purpose='Localize already-rejected TR8 byte differences; compare unchanged-parent repeats, graph output buffers, independent scratch, and intermediate/norm/final values.',
        selection='None. No performance comparison, no algorithm retry, no scored upload.',
        generations='ctr[3] advances once; active ss bank=(new_generation-1)&1; valid expert-major route prefix only.',
        GPU_jobs=0, max_checks=1, matrix='M1,3,8,16,32 x 5 routing fixtures x 3 input seeds x 2 cycles x 6 serial calls; TR8 only <=8, larger D is another parent graph.',
        added_scratch='One parent-equivalent independent scratch bank during prepare; released before official calls.',
        stop='Stop starting probes at 278 s; parent timed path retained regardless of diagnostic outcome.')
    if args.compact:
        record.update(reporting_repair_only=True, diagnosed_defect='d563fa2d output exceeded the 16000-character retained log tail',
                      final_summary_z64_limit=9000, raw_records_z64_limit=4000,
                      additional_change='Diagnostic error metrics use FP64 to avoid report overflow; timed math unchanged.')
    (out / 'binding.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))


if __name__ == '__main__': main()
