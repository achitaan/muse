"""Pin every inline build without changing any embedded kernel or math flags."""
import ast


HELPERS = '''\n# Explicit flags, not process-global TORCH_CUDA_ARCH_LIST: builders run in parallel.
def _gate_host_flags(flags=()):
    return [f for f in flags if not f.startswith(('-std=', '--std='))] + ['-std=c++20']


def _gate_nvcc_flags(flags=()):
    return [f for f in flags if not f.startswith(('-std=', '--std=', '-gencode',
            '--generate-code', '-arch=', '--gpu-architecture='))] + [
            '-std=c++20', '-gencode=arch=compute_90a,code=sm_90a']

'''


def pin_inline_builds(source):
    """Source-span edits preserve readable source and all incumbent non-ISA flags."""
    tree = ast.parse(source)
    raw = source.encode()
    starts = [0]
    for line in raw.splitlines(keepends=True):
        starts.append(starts[-1] + len(line))

    def span(node):
        return starts[node.lineno - 1] + node.col_offset, starts[node.end_lineno - 1] + node.end_col_offset

    edits = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in ('load_inline', '_tvm_load_inline')):
            continue
        kws = {k.arg: k for k in node.keywords}
        missing = []
        for name, helper, default in (('extra_cflags', '_gate_host_flags', "['-O3']"),
                                      ('extra_cuda_cflags', '_gate_nvcc_flags', "['-O3']")):
            if name in kws:
                a, b = span(kws[name].value)
                edits.append((a, b, helper.encode() + b'(' + raw[a:b] + b')'))
            else:
                missing.append(f'{name}={helper}({default})')
        if missing:
            _, end = span(node)
            comma = '' if raw[:end - 1].rstrip().endswith(b',') else ', '
            edits.append((end - 1, end - 1, (comma + ', '.join(missing)).encode()))

    # Keep the module docstring and mandatory okc header in their original place.
    first = tree.body[1] if isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant) else tree.body[0]
    at = starts[first.lineno - 1]
    edits.append((at, at, HELPERS.encode()))
    for a, b, replacement in sorted(edits, reverse=True):
        raw = raw[:a] + replacement + raw[b:]
    result = raw.decode()
    ast.parse(result)
    return result
