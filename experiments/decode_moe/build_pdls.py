"""Lean T1/T8 early-norm candidate; preserve both earlier CPU drafts."""

import ast
import hashlib
import json

from pdl16_sources import HERE, ROOT, BASE_SHA
from pdl_norm_sources import UPSTREAM, NOTICE, QUERY, pdln_derive

OUT = ROOT / 'results/decode-moe-plan-20261030/pdl-norm-small'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw = (HERE / 'decoder_pdli.py').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == BASE_SHA
    source = raw.decode()
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_pdli_layer')
    old = ast.get_source_segment(source, node)
    expr = "state['attn_boundary'](data['stream'], attn_output, branch_dtype=torch.bfloat16, router_dtype=None if split_router else torch.float32)"
    assert old.count(expr) == 1
    layer = old.replace('def _pdli_layer(', 'def _pdls_layer(', 1).replace(expr, "_pdls_boundary(data['stream'], attn_output, state)")
    helper = (HERE / 'pdl_norm_sources.py').read_text()
    node = next(n for n in ast.parse(helper).body if isinstance(n, ast.FunctionDef) and n.name == 'pdln_derive')
    helper = ast.get_source_segment(helper, node)
    query = QUERY.replace('<8192,256,true,true,false,false,false,true>',
                          '<8192,256,true,true,false,true,false,true>')
    assert query != QUERY and query.replace('<8192,256,true,true,false,true,false,true>',
                                             '<8192,256,true,true,false,false,false,true>') == QUERY
    norm = (UPSTREAM / 'norm_residual.cuh').read_text()
    cuda = NOTICE + pdln_derive(norm) + query
    pin = json.loads((OUT.parent / 'pdl-norm/source-binding.json').read_text())
    headers = dict(pin['canonical_LF_hashes'])
    headers['runtime.cuh'] = hashlib.sha256((UPSTREAM / 'runtime.cuh').read_text().encode()).hexdigest()
    sha = hashlib.sha256(cuda.encode()).hexdigest()
    append = '\n# Unvalidated T1/T8 early norm; all original arithmetic and byte gates retained.\n'
    append += helper + '\n' + layer + '\n'
    for key, value in {'_PDLS_HEADERS': headers, '_PDLS_QUERY': query,
                       '_PDLS_NOTICE': NOTICE, '_PDLS_SOURCE_SHA': sha}.items():
        append += key + ' = ' + repr(value) + '\n'
    append += (HERE / 'pdls_runtime.py').read_text()
    blob = raw + append.encode()
    ast.parse(blob)
    assert blob.startswith(raw) and len(blob) < 1_000_000
    (HERE / 'decoder_pdls_draft.py').write_bytes(blob)
    (OUT / 'norm-router-early.cuh').write_bytes(cuda.encode())
    record = dict(candidate_sha256=hashlib.sha256(blob).hexdigest(), bytes=len(blob),
        headroom=1_000_000-len(blob), scored_parent_sha256=BASE_SHA,
        full_scored_parent_prefix_identical=True, private_CUDA_sha256=sha,
        installed_header_hashes=headers,
        HasRouter=True, pre_norm_bf16=False, selected_domain='B1,L1 or L8,H8192',
        private_projection_builds=0, new_SDK_host_instantiations=1,
        gate='Exact norm/router/R2f unit bytes and >3% both orders; then complete layer/cache bytes and >3% both orders. One union, no sweep.',
        GPU_jobs=0, GPU_admitted=False)
    (OUT / 'candidate-binding.json').write_bytes((json.dumps(record, indent=2)+'\n').encode())
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
