"""Build the private early-trigger norm paired with the 16/32 FPR path."""

import ast
import hashlib
import json

from pdl16_sources import HERE, BASE_SHA
from pdl_norm_sources import OUT, QUERY, NOTICE, main as derive

CHAIN_SHA = '985abda3f1a91e6daca07a38e9f91a397b4942ef892285704ddbd1792bff9a40'


def main():
    derive()
    raw = (HERE / 'decoder_pdl_chain_draft.py').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == CHAIN_SHA
    src = raw.decode()
    parent = (HERE / 'decoder_pdli.py').read_bytes()
    assert hashlib.sha256(parent).hexdigest() == BASE_SHA and raw.startswith(parent)
    tree = ast.parse(src)
    targets = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    edits = {}
    for name in ['_p16_layer', '_p16_select']:
        old = ast.get_source_segment(src, targets[name])
        new = old
        if name == '_p16_layer':
            a = "state['attn_boundary'](data['stream'], attn_output, branch_dtype=torch.bfloat16, router_dtype=None if split_router else torch.float32)"
            b = "(_pdln_boundary(data['stream'], attn_output, state) if bits & 2 else " + a + ')'
            assert new.count(a) == 1
            new = new.replace(a, b)
        else:
            a = "if state.get('fpr') is not None and _okh_budget(start, 274.):"
            b = "if state.get('fpr') is not None and state.get('pdln', {}).get('ready') and _okh_budget(start, 274.):"
            assert new.count(a) == 1
            new = new.replace(a, b)
            a = "state['attn_boundary'](stream, ao, branch_dtype=torch.bfloat16, router_dtype=None)"
            b = "(_pdln_boundary(stream, ao, state) if on else " + a + ')'
            assert new.count(a) == 1
            new = new.replace(a, b)
        assert src.count(old) == 1
        src = src.replace(old, new)
        edits[name] = dict(old_sha256=hashlib.sha256(old.encode()).hexdigest(),
                           new_sha256=hashlib.sha256(new.encode()).hexdigest())
    binding = json.loads((OUT / 'source-binding.json').read_text())
    helper_tree = ast.parse((HERE / 'pdl_norm_sources.py').read_text())
    helper_node = next(n for n in helper_tree.body if isinstance(n, ast.FunctionDef) and n.name == 'pdln_derive')
    helper = ast.get_source_segment((HERE / 'pdl_norm_sources.py').read_text(), helper_node)
    append = '\n# Unvalidated private SDK norm; its FPR pair shares all layer gates.\n'
    append += helper + '\n_PDLN_EXPECTED_HEADERS = ' + repr(binding['canonical_LF_hashes']) + '\n'
    append += '_PDLN_QUERY = ' + repr(QUERY) + '\n'
    append += '_PDLN_NOTICE = ' + repr(NOTICE) + '\n'
    append += '_PDLN_SOURCE_SHA = ' + repr(binding['source_with_query_sha256']) + '\n'
    append += (HERE / 'pdl_norm_runtime.py').read_text()
    blob = src.encode() + append.encode()
    ast.parse(blob)
    assert blob.startswith(parent) and len(blob) < 1_000_000
    (HERE / 'decoder_pdln_draft.py').write_bytes(blob)
    record = dict(status='CPU derivation; GPU-unvalidated, not admitted',
                  candidate_sha256=hashlib.sha256(blob).hexdigest(), bytes=len(blob),
                  headroom=1_000_000-len(blob), scored_parent_sha256=BASE_SHA,
                  CPU_chain_parent_sha256=CHAIN_SHA, full_scored_parent_prefix_identical=True,
                  changed_functions=edits, norm_source_sha256=binding['source_with_query_sha256'],
                  shared_gate='bit2 now denotes private norm + FPR; unit bytes, then whole-layer bytes and >3% both orders.',
                  GPU_jobs=0, prepare_time_unverified=True)
    (OUT / 'candidate-binding.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
