"""Build the four separately inspectable candidate submissions from the retained best/ files.

CPU only. Writes experiments/opus_fresh/candidates/{moe_finreg,decoder_finreg,dense_o128,decoder_o128}.py
and manifest.json. Never modifies best/ or candidates/.
"""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
import sys
sys.path.insert(0, str(HERE))
import finreg_edits  # noqa: E402

LIMIT = 1_000_000  # service source-size limit recorded in experiments/prepared/decoder_moe_sliced_review.md

DENSE_WRAP = '''

_o128_base_prepare = prepare
_o128_base_kernel = kernel_fn


def prepare(weights):
    started = _o128_time.monotonic()
    box = _o128_start()
    state = _o128_base_prepare(weights)
    try:
        state['o128'] = _o128_select(box, weights, lambda x, layer: _o128_base_kernel(
            {'x': x, 'config': {'m': x.shape[0], 'n': weights[layer].shape[0], 'k': x.shape[1], 'layer': layer}}, state), started, 250.0)
    except Exception as e:
        print('O128_ERROR', repr(e), flush=True)
        state['o128'] = {}
    return state


def kernel_fn(data, state):
    y = _o128_run(state.get('o128'), data['x'], data['config']['layer'], data['config']['n'])
    return _o128_base_kernel(data, state) if y is None else y
'''

DECODER_WRAP = '''

_o128_dec_prepare = prepare
_o128_dec_project = _achita_dense_project


def prepare(weights):
    started = _o128_time.monotonic()
    box = _o128_start()
    state = _o128_dec_prepare(weights)
    state['o128'] = {}
    try:
        state['o128'] = _o128_select(box, weights, lambda x, layer: _o128_dec_project(x, layer, state), started, 235.0)
    except Exception as e:
        print('O128_ERROR', repr(e), flush=True)
    return state


def _achita_dense_project(x, layer, state, rows=None):
    y = None if rows is not None else _o128_run(state.get('o128'), x, layer, state['dense'][layer].shape[0])
    return _o128_dec_project(x, layer, state, rows) if y is None else y
'''


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def replace_const(src: str, name: str, fn):
    tree = ast.parse(src)
    node = next(n for n in tree.body if isinstance(n, ast.Assign) and getattr(n.targets[0], 'id', '') == name)
    seg = ast.get_source_segment(src, node.value)
    assert seg == repr(node.value.value) and src.count(seg) == 1
    new_val = fn(node.value.value)
    return src.replace(seg, repr(new_val)), node.value.value, new_val


def ast_dump_without(src: str, const_name: str):
    tree = ast.parse(src)
    for n in tree.body:
        if isinstance(n, ast.Assign) and getattr(n.targets[0], 'id', '') == const_name:
            n.value = ast.Constant('<CUDA>')
    return ast.dump(tree)


def build_finreg(parent: Path, const: str, out: Path, old_name: str, new_name: str):
    src = open(parent, encoding='utf-8', newline='').read()
    new_src, old_cuda, new_cuda = replace_const(src, const, finreg_edits.apply)
    assert finreg_edits.invert(new_cuda) == old_cuda
    assert new_src.count(old_name) == 1
    new_src = new_src.replace(old_name, new_name)
    out.write_text(new_src, encoding='utf-8', newline='')
    # inverse: restore exact parent bytes
    back = new_src.replace(new_name, old_name)
    back, _, _ = replace_const(back, const, finreg_edits.invert)
    assert back == src, 'inverse edit does not restore the parent'
    # only the CUDA constant and the extension-name literal differ in the module AST
    assert ast_dump_without(new_src.replace(new_name, old_name), const) == ast_dump_without(src, const)
    return {'parent': str(parent.relative_to(ROOT)), 'parent_sha256': sha(src.encode()),
            'candidate': str(out.relative_to(ROOT)), 'candidate_sha256': sha(new_src.encode()),
            'bytes': len(new_src.encode()), 'under_size_limit': len(new_src.encode()) <= LIMIT,
            'cuda_parent_sha256': sha(old_cuda.encode()), 'cuda_candidate_sha256': sha(new_cuda.encode()),
            'edits': ['wgs_flush DF branch: release-only', 'wgs_seg split tail: in-register designated finisher',
                      f'extension name {old_name} -> {new_name}'],
            'inverse_restores_parent_exactly': True, 'ast_outside_cuda_and_name_unchanged': True}


def build_o128(parent: Path, out: Path, wrap: str):
    src = open(parent, encoding='utf-8', newline='').read()
    common = (HERE / 'o128_common.py').read_text(encoding='utf-8')
    eol = '\r\n' if '\r\n' in src else '\n'
    new_src = src + (common + wrap).replace('\n', eol)
    out.write_text(new_src, encoding='utf-8', newline='')
    assert new_src.startswith(src) and src.endswith(eol)
    ast.parse(new_src)
    return {'parent': str(parent.relative_to(ROOT)), 'parent_sha256': sha(src.encode()),
            'candidate': str(out.relative_to(ROOT)), 'candidate_sha256': sha(new_src.encode()),
            'bytes': len(new_src.encode()), 'under_size_limit': len(new_src.encode()) <= LIMIT,
            'parent_is_exact_prefix': True, 'appended': 'o128_common.py + wrapper'}


def main():
    best = ROOT / 'best'
    man = {
        'moe_finreg': build_finreg(best / 'moe_verified.py', 'CUDA_SRC', HERE / 'moe_finreg.py',
                                   'okc_moe_w4a16_v216x_pair1dq8_', 'okc_moe_w4a16_v216x_finreg1_'),
        'decoder_finreg': build_finreg(best / 'decoder_verified.py', '_native_moe_CUDA_SRC', HERE / 'decoder_finreg.py',
                                       'okc_moe_w4a16_v216x_pair1dq8_', 'okc_moe_w4a16_v216x_finreg1_'),
        'dense_o128': build_o128(best / 'verified.py', HERE / 'dense_o128.py', DENSE_WRAP),
        'decoder_o128': build_o128(best / 'decoder_verified.py', HERE / 'decoder_o128.py', DECODER_WRAP),
        'gpu_status': 'unmeasured: no GPU job, official check or score was run',
    }
    (HERE / 'manifest.json').write_text(json.dumps(man, indent=2), encoding='utf-8')
    print(json.dumps(man, indent=2))


if __name__ == '__main__':
    main()
