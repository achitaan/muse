"""Exact integration inverses, immutable fallbacks and source binding audit."""

import ast
import hashlib
import json

from pdl_norm_sources import HERE, OUT


def main():
    a = (HERE / 'decoder_pdl_chain_draft.py').read_text()
    b = (HERE / 'decoder_pdln_draft.py').read_text()
    funcs = lambda s:{n.name:n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)}
    af, bf = funcs(a), funcs(b)
    old = ast.get_source_segment(a,af['_p16_layer'])
    new = ast.get_source_segment(b,bf['_p16_layer'])
    expr = "state['attn_boundary'](data['stream'], attn_output, branch_dtype=torch.bfloat16, router_dtype=None if split_router else torch.float32)"
    patch = "(_pdln_boundary(data['stream'], attn_output, state) if bits & 2 else "+expr+")"
    assert new.count(patch)==1 and new.replace(patch,expr)==old
    old = ast.get_source_segment(a,af['_p16_select'])
    new = ast.get_source_segment(b,bf['_p16_select'])
    new = new.replace("if state.get('fpr') is not None and state.get('pdln', {}).get('ready') and _okh_budget(start, 274.):", "if state.get('fpr') is not None and _okh_budget(start, 274.):")
    expr = "state['attn_boundary'](stream, ao, branch_dtype=torch.bfloat16, router_dtype=None)"
    new = new.replace("(_pdln_boundary(stream, ao, state) if on else "+expr+")",expr)
    assert new==old
    for name in ['kernel_fn','_p16_project','_p16_unpool','_p16_pool_route']:
        assert ast.dump(af[name],include_attributes=False)==ast.dump(bf[name],include_attributes=False)
    raw = (HERE / 'decoder_pdln_draft.py').read_bytes()
    assert raw.startswith((HERE / 'decoder_pdli.py').read_bytes()) and len(raw)<1_000_000
    binding = json.loads((OUT / 'candidate-binding.json').read_text())
    assert hashlib.sha256(raw).hexdigest()==binding['candidate_sha256']
    record = dict(GPU_jobs=0,layer_inverse_exact=True,selector_inverse_exact=True,
                  public_dispatch_and_O_U_FPR_AST_unchanged=True,
                  full_scored_parent_prefix_identical=True,
                  candidate_sha256=binding['candidate_sha256'],bytes=len(raw),
                  limitation='Static integration/source proof; no H200 build, numerical bytes, overlap or performance validation.')
    (OUT / 'integration-audit.json').write_bytes((json.dumps(record,indent=2)+'\n').encode())
    print(json.dumps(record,indent=2))


if __name__ == '__main__': main()
