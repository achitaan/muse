"""Build the bounded O/FPR/unpooler draft from the unchanged scored parent."""

import ast
import base64
import hashlib
import json
import zlib

from pdl16_sources import derive as derive_o, parents as parents_o, HERE, ROOT, M16_PREFIX, M16_INST, BASE_SHA
from pdlu_sources import derive as derive_u, U16_INST, U32_INST

OUT = ROOT / 'results/decode-moe-plan-20261030/pdl-chain'


def main():
    oo = derive_o()
    _, ocpp = parents_o()
    uu, ucpp = derive_u()
    source_parts = {**oo, **uu}
    packed = '\n'.join(v[1] for v in source_parts.values())
    cpp = '\n'.join([ocpp['o16'], ocpp['o32'], 'int pdlo32_info(int);', ucpp['u16'], ucpp['u32']])
    functions = [M16_PREFIX + '_launch', 'ol_launch', 'pdlo32_info', 'dd_cap2_b_cpssm_launch', 'launch', 'info', 'pdlu32_occupancy']
    raw = (HERE / 'decoder_pdli.py').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == BASE_SHA
    text = raw.decode()
    tr = ast.parse(text)
    fn = next(n for n in tr.body if isinstance(n, ast.FunctionDef) and n.name == '_pdli_layer')
    layer = old = ast.get_source_segment(text, fn)
    edits = [
        ('def _pdli_layer(data, state, mask):', 'def _p16_layer(data, state, bits):\n mask = 0'),
        ("_pdli_project(o, 'o', state) if mask & 2 else _achita_dense_project(o, 'o', state)",
         "_p16_project(o, state) if bits & 1 else _achita_dense_project(o, 'o', state)"),
        ("_fpr_bfp_fused_pooler_router(ffn_input, state['fpr'])",
         "(_p16_pool_route(ffn_input, state) if bits & 2 else _fpr_bfp_fused_pooler_router(ffn_input, state['fpr']))"),
        ("_achita_dense_project(expert_output, 'unpooler', state)",
         "(_p16_unpool(expert_output, state) if bits & 4 else _achita_dense_project(expert_output, 'unpooler', state))"),
    ]
    for a, b in edits:
        assert layer.count(a) == 1
        layer = layer.replace(a, b)
    inverse = layer
    for a, b in reversed(edits):
        assert inverse.count(b) == 1
        inverse = inverse.replace(b, a)
    assert inverse == old
    append = '\n# Unvalidated 16/32 Decode handoff bundle; no new path is enabled without gates.\n'
    append += '_P16_CUDA = _zsrc(' + repr(base64.b64encode(zlib.compress(packed.encode(), 9)).decode()) + ')\n'
    append += '_P16_CUDA_SHA = ' + repr(hashlib.sha256(packed.encode()).hexdigest()) + '\n'
    append += '_P16_CPP = ' + repr(cpp) + '\n_P16_FUNCTIONS = ' + repr(functions) + '\n'
    append += '_P16_M16_EXPORT = ' + repr(M16_PREFIX + '_launch') + '\n'
    append += '_P16_M16_INST = ' + repr(M16_INST) + '\n'
    append += '_P16_U16_INST = ' + repr(U16_INST) + '\n'
    append += layer + '\n' + (HERE / 'pdl_chain_runtime.py').read_text()
    blob = raw + append.encode()
    ast.parse(blob)
    assert blob.startswith(raw) and len(blob) < 1_000_000
    (HERE / 'decoder_pdl_chain_draft.py').write_bytes(blob)
    OUT.mkdir(exist_ok=True)
    (OUT / 'combined.cu').write_bytes(packed.encode())
    (OUT / 'combined.cpp').write_bytes(cpp.encode())
    record = dict(status='CPU-prepared, unvalidated; no GPU admission yet', parent_sha256=BASE_SHA,
        candidate_sha256=hashlib.sha256(blob).hexdigest(), bytes=len(blob), headroom=1_000_000-len(blob),
        full_parent_prefix_identical=True, layer_inverse_exact=True,
        combined_CUDA_sha256=hashlib.sha256(packed.encode()).hexdigest(),
        CPP_sha256=hashlib.sha256(cpp.encode()).hexdigest(),
        runtime_sha256=hashlib.sha256((HERE / 'pdl_chain_runtime.py').read_bytes()).hexdigest(),
        layer_sha256=hashlib.sha256(layer.encode()).hexdigest(),
        source_parts={k: v[2]['candidate_sha256'] for k, v in source_parts.items()},
        FPR='Existing compiled kernel, fixed full-ring depths 5/3; reset host launch flag to zero after each launch.',
        gates='Unit bytes on changing inputs, then one eligible union per regime; whole-layer bytes and >3% both orders.',
        GPU_jobs=0, preparation_time_unverified=True)
    (OUT / 'binding.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
