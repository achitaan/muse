"""Prepare, without evaluating, native MoE/attention integration in the decoder."""
import ast
import hashlib
import json
from pathlib import Path

root = next(p for p in Path(__file__).resolve().parents if (p / ".git").exists())

def namespace(source, prefix):
    tree = ast.parse(source)
    symbols = set()
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            symbols.add(n.name)
        elif isinstance(n, ast.Assign):
            symbols.update(t.id for t in n.targets if isinstance(t, ast.Name))
    names = {name: prefix + name for name in symbols}
    class Rename(ast.NodeTransformer):
        def visit_Name(self, node):
            node.id = names.get(node.id, node.id)
            return node
        def visit_FunctionDef(self, node):
            node.name = names.get(node.name, node.name)
            return self.generic_visit(node)
        def visit_ClassDef(self, node):
            node.name = names.get(node.name, node.name)
            return self.generic_visit(node)
    strings_before = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant)
                      and isinstance(n.value, str) and '#include' in n.value]
    tree = Rename().visit(tree)
    ast.fix_missing_locations(tree)
    text = ast.unparse(tree)
    strings_after = [n.value for n in ast.walk(ast.parse(text)) if isinstance(n, ast.Constant)
                     and isinstance(n.value, str) and '#include' in n.value]
    assert strings_before == strings_after
    return text

moe = (root / '.tools/native-moe-prepared.py').read_text()
attn = (root / 'best/attention_checked.py').read_text()
decoder = (root / 'best/decoder_verified.py').read_text()
# Only the original setup uses this loader. The new native expert path owns
# its lossless packed weights/scratch; no second Marlin bank is needed.
t = ast.parse(decoder)
bank = next(n for n in t.body if isinstance(n, ast.FunctionDef) and n.name == '_marlin_bank')
lines = decoder.splitlines(keepends=True)
lines[bank.lineno - 1:bank.end_lineno] = ['def _marlin_bank(weights):\n    return None\n']
decoder = ''.join(lines)
old = '''    out = _achita_decoder_attention(data, qkv)
    gate = qkv[:, (hq + 2 * hkv) * d:]
    o = state["o_norm_gate"](out.reshape(tokens, hq * d).to(torch.bfloat16), gate)'''
new = '''    attention_data = dict(data, qkv=qkv.reshape(batch, length, -1))
    o = _native_attn_kernel_fn(attention_data).reshape(tokens, hq * d)'''
assert decoder.count(old) == 1
decoder = decoder.replace(old, new)

wrapper = '''

_achita_before_native_prepare = prepare

def prepare(weights):
    state = _achita_before_native_prepare(weights)
    expert_weights = {k: weights[k] for k in
        ("w13", "w13_sf", "w13_gs", "w2", "w2_sf", "w2_gs")}
    expert_weights["post_norm_w"] = weights["post_expert_norm"]
    state["native_moe"] = _native_moe_prepare(expert_weights)
    return state

def _experts(x, topk_weights, topk_ids, state):
    # The caller already rotates this rank's local experts to IDs 0..31.
    return _native_moe_kernel_fn({"hidden_states": x, "topk_ids": topk_ids,
        "topk_weights": topk_weights,
        "config": {"local_start": 0, "local_count": 32}}, state["native_moe"])
'''
license_text = (root / '.tools/vjeux-LICENSE').read_text()
notice = ('# Native attention/MoE derived from Vjeux, okc-2026, Apache-2.0,\n'
          '# revision f35a41ee1c13e316dabbf772830d47b23b453393.\n'
          '# Local changes: namespacing, decoder integration, cache lifetime fixes,\n'
          '# BF16 MoE boundaries; shared scratch requires sequential replay.\n' +
          '\n'.join('# ' + x for x in license_text.splitlines()) + '\n')
combined = ('# okc: problem=muse_decode_layer\n' + notice + '\n' +
            namespace(moe, '_native_moe_') + '\n\n' +
            namespace(attn, '_native_attn_') + '\n\n' + decoder + wrapper)
assert 'F.linear(router_input, state["router"])' in combined
assert 'logits = torch.mm(ffn_input, state["router_hi_t"], out_dtype=torch.float32)' in combined
assert 'MIT License' in combined and 'Apache License' in combined
compile(combined, '<native-decoder-draft>', 'exec')
out = root / '.tools/decode-native-variant.py'
out.write_bytes(combined.encode())
audit = {'source_sha256': hashlib.sha256(combined.encode()).hexdigest(),
         'moe_sha256': hashlib.sha256(moe.encode()).hexdigest(),
         'attention_sha256': hashlib.sha256(attn.encode()).hexdigest(),
         'decoder_base_sha256': hashlib.sha256((root / 'best/decoder_verified.py').read_bytes()).hexdigest(),
         'cuda_string_payloads_preserved': True, 'fp32_router_path_preserved': True,
         'moe_local_ids': 'caller rotates expert IDs; native owns 0..31',
         'new_attention_cache_entries': 'same input cache tensors passed to native kernel',
         'gpu_status': 'unmeasured'}
(root / '.tools/decode-native-audit.json').write_text(json.dumps(audit, indent=2))
print(json.dumps(audit))
