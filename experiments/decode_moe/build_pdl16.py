"""Build a gated, unvalidated Decode O16/O32 draft without editing its parent."""

import ast
import base64
import hashlib
import json
import zlib

from pdl16_sources import derive, parents, HERE, OUT, M16_PREFIX, M16_INST, BASE_SHA


def main():
    sources = derive()
    _, cpp = parents()
    packed = '\n'.join(v[1] for v in sources.values())
    cpp_source = cpp['o16'] + '\n' + cpp['o32'] + '\nint pdlo32_info(int);\n'
    raw = (HERE / 'decoder_pdli.py').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == BASE_SHA
    text = raw.decode()
    tr = ast.parse(text)
    fn = next(n for n in tr.body if isinstance(n, ast.FunctionDef) and n.name == '_pdli_layer')
    old = ast.get_source_segment(text, fn)
    new = old.replace('def _pdli_layer(data, state, mask):', 'def _p16_layer(data, state):\n mask = 0', 1)
    a = "_pdli_project(o, 'o', state) if mask & 2 else _achita_dense_project(o, 'o', state)"
    b = '_p16_project(o, state)'
    assert new.count(a) == 1
    new = new.replace(a, b)
    inverse = new.replace(b, a).replace('def _p16_layer(data, state):\n mask = 0', 'def _pdli_layer(data, state, mask):', 1)
    assert inverse == old
    extra = '\n# Unvalidated O16/O32 PDL draft; defaults to the exact scored parent.\n'
    extra += '_P16_CUDA = _zsrc(' + repr(base64.b64encode(zlib.compress(packed.encode(), 9)).decode()) + ')\n'
    extra += '_P16_CUDA_SHA = ' + repr(hashlib.sha256(packed.encode()).hexdigest()) + '\n'
    extra += '_P16_CPP = ' + repr(cpp_source) + '\n'
    extra += '_P16_FUNCTIONS = ' + repr([M16_PREFIX + '_launch', 'ol_launch', 'pdlo32_info']) + '\n'
    extra += '_P16_M16_EXPORT = ' + repr(M16_PREFIX + '_launch') + '\n'
    extra += '_P16_M16_INST = ' + repr(M16_INST) + '\n'
    extra += new + '\n' + (HERE / 'pdl16_runtime.py').read_text()
    candidate = raw + extra.encode()
    ast.parse(candidate)
    assert candidate.startswith(raw) and len(candidate) < 1_000_000
    (HERE / 'decoder_pdl16_draft.py').write_bytes(candidate)
    (OUT / 'combined.cu').write_bytes(packed.encode())
    (OUT / 'combined.cpp').write_bytes(cpp_source.encode())
    record = dict(status='CPU-prepared draft; not GPU-checked or published', parent_sha256=BASE_SHA,
        candidate_sha256=hashlib.sha256(candidate).hexdigest(), bytes=len(candidate),
        headroom=1_000_000-len(candidate), full_parent_prefix_identical=True,
        layer_edit_exact_inverse=True, combined_CUDA_sha256=hashlib.sha256(packed.encode()).hexdigest(),
        changes='O16/O32 projection handoffs only; other arithmetic/fallbacks and small-token PDL retained.',
        unimplemented='16/32 unpooler PDL and existing fused-pooler incoming PDL are not enabled.',
        GPU_jobs=0, GPU_preparation_time_unverified=True)
    (OUT / 'draft-binding.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
