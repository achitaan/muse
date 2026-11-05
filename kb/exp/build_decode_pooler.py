"""Assemble the self-contained check file; no GPU or third-party packages needed."""
import ast
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / 'v1' / 'decode_v1.py'


def assemble():
    source = BASE.read_text()
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    edits = []
    # Explicit per-build flags, including builders imported inside functions and
    # inside the embedded Chen module. Do not mutate process-global env/modules.
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id not in ('load_inline', '_tvm_load_inline'):
            continue
        present = set()
        for kw in node.keywords:
            if kw.arg not in ('extra_cflags', 'extra_cuda_cflags'):
                continue
            present.add(kw.arg)
            lo = offsets[kw.value.lineno - 1] + kw.value.col_offset
            hi = offsets[kw.value.end_lineno - 1] + kw.value.end_col_offset
            helper = '_dp_nvcc_flags' if kw.arg == 'extra_cuda_cflags' else '_dp_host_flags'
            edits.append((lo, hi, helper + '(' + source[lo:hi] + ')'))
        at = offsets[node.end_lineno - 1] + node.end_col_offset - 1
        additions = []
        for name, helper in (('extra_cflags', '_dp_host_flags'), ('extra_cuda_cflags', '_dp_nvcc_flags')):
            if name not in present:
                additions.append(f'{name}={helper}(["-O3"])')
        if additions:
            # Some multiline calls already have a trailing comma.
            prefix = ' ' if source[:at].rstrip().endswith(',') else ', '
            edits.append((at, at, prefix + ', '.join(additions)))
    for lo, hi, value in sorted(edits, reverse=True):
        source = source[:lo] + value + source[hi:]
    helpers = '''# Explicit CUDA 13 / SM90a build flags for every inherited builder.
def _dp_host_flags(flags=()):
    return [f for f in flags if not f.startswith(('-std=', '--std='))] + ['-std=c++20']

def _dp_nvcc_flags(flags=()):
    return [f for f in flags if not f.startswith(('-std=', '--std=', '-gencode',
            '--generate-code', '-arch=', '--gpu-architecture='))] + [
            '-std=c++20', '-gencode=arch=compute_90a,code=sm_90a']

'''
    marker = 'import os\n'
    assert source.count(marker) >= 1
    source = source.replace(marker, helpers + marker, 1)
    source += '\n\n' + (HERE / 'decode_pooler_tail.py').read_text()
    ast.parse(source)
    assert len(source.encode()) < 1_000_000, len(source.encode())
    return source


if __name__ == '__main__':
    path = HERE / 'decode_pooler.py'
    path.write_text(assemble())
    print(path, path.stat().st_size)
