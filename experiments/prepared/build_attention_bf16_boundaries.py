"""Restore pinned SGLang's BF16 boundaries without changing attention dispatch."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'STATE.json').is_file())
BASE_SHA = 'b56e07b1dbc930ef8fd4adaeba108e6d17b62a0f390d68c024b98d3cd608aee4'
SOURCE = ROOT / '.attention-measurements' / BASE_SHA / 'submission.py'
if not SOURCE.exists(): SOURCE = ROOT / 'best/attention_checked.py'


def literal(source, name):
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))
    return node, ast.literal_eval(node.value)


def corrected_cuda(old):
    changes = []
    def change(before, after, count):
        nonlocal old
        assert old.count(before) == count, (before, old.count(before), count)
        old = old.replace(before, after)
        changes.append((before, after, count))
    change('// conformance sensitivity probe only : round the attention output, which SGLang keeps in\n'
           '// BF16 before rmsnorm * gate, to e4m3 -- the precision check must reject that build\n'
           '__device__ __forceinline__ float pr8(float x) {\n  return x;\n}',
           '// Pinned SGLang materializes BF16 Q/K and BF16 attention output.\n'
           '__device__ __forceinline__ float bf16_round(float x) {\n'
           '  return __bfloat162float(__float2bfloat16_rn(x));\n}\n'
           '__device__ __forceinline__ float pr8(float x) {\n'
           '  return bf16_round(x);\n}', 1)
    # Five output finalizers, with 4 + 8 + 1 + 1 + 1 scalar call sites.
    change('pr8(', 'attn_out_bf16(', 16)
    # QDIRECT Q[16], regular Q[8], and new K[8]. RoPE consumes BF16 norm.
    change('x[e] *= r;', 'x[e] = bf16_round(x[e] * r);', 3)
    # RoPE outputs are stored as BF16 before FP8 Q/cache conversion.
    change('x[2 * e] = a0 * c - a1 * sn;',
           'x[2 * e] = bf16_round(a0 * c - a1 * sn);', 3)
    change('x[2 * e + 1] = a0 * sn + a1 * c;',
           'x[2 * e + 1] = bf16_round(a0 * sn + a1 * c);', 3)
    return old, changes


def build():
    raw = SOURCE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == BASE_SHA
    text = raw.decode('utf8')
    node, old = literal(text, 'CUDA_SRC')
    new, changes = corrected_cuda(old)
    lines = text.splitlines(keepends=True)
    start = sum(map(len, lines[:node.value.lineno - 1])) + node.value.col_offset
    end = sum(map(len, lines[:node.value.end_lineno - 1])) + node.value.end_col_offset
    result = text[:start] + repr(new) + text[end:]
    result = result.replace('# CUDA algorithm is unchanged.',
        '# Local CUDA changes restore BF16 Q/K norm, RoPE and attention-output boundaries.')
    fallback_before = '''    q = _rms(q_in.reshape(batch, length, hq, d)) * cfg["qk_scale_factor"]
    k = _rms(k_in.reshape(batch, length, hkv, d))
    if cfg["rope_theta"]:
        q, k = _rope(q, pos, cfg["rope_theta"]), _rope(k, pos, cfg["rope_theta"])
    q = q.to(torch.bfloat16)'''
    fallback_after = '''    q = _rms(q_in.reshape(batch, length, hq, d)).to(torch.bfloat16).float()
    k = _rms(k_in.reshape(batch, length, hkv, d)).to(torch.bfloat16).float()
    if cfg["rope_theta"]:
        q, k = _rope(q, pos, cfg["rope_theta"]), _rope(k, pos, cfg["rope_theta"])
    q, k = q.to(torch.bfloat16), k.to(torch.bfloat16)'''
    assert result.count(fallback_before) == 1
    result = result.replace(fallback_before, fallback_after)
    scale_before = '* cfg["sm_scale"]\n'
    scale_after = '* (cfg["sm_scale"] * cfg["qk_scale_factor"])\n'
    assert result.count(scale_before) == 1
    result = result.replace(scale_before, scale_after)
    assert literal(result, 'CUDA_SRC')[1] == new
    reverse = new
    for before, after, count in reversed(changes):
        assert reverse.count(after) == count
        reverse = reverse.replace(after, before)
    assert reverse == old
    original_tree = ast.parse(text)
    result_tree = ast.parse(result)
    next(n for n in result_tree.body if isinstance(n, ast.Assign)
         and any(isinstance(t, ast.Name) and t.id == 'CUDA_SRC' for t in n.targets)).value = node.value
    new_fallback = next(n for n in result_tree.body if isinstance(n, ast.FunctionDef) and n.name == '_fallback')
    old_fallback = next(n for n in original_tree.body if isinstance(n, ast.FunctionDef) and n.name == '_fallback')
    result_tree.body[result_tree.body.index(new_fallback)] = old_fallback
    assert ast.dump(original_tree) == ast.dump(result_tree)
    out = ROOT / '.tools/attention-bf16-boundaries-variant.py'
    out.write_bytes(result.encode('utf8'))
    report = dict(parent_sha256=BASE_SHA, source_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
                  bytes=out.stat().st_size, inverse_exact=True, other_ast_unchanged=True,
                  changed_python_functions=['_fallback'],
                  replacements=[dict(before=a, after=b, count=n) for a,b,n in changes],
                  pinned_sglang='a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f')
    (ROOT / '.tools/attention-bf16-boundaries-source-audit.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps({k:v for k,v in report.items() if k!='replacements'},indent=2))


if __name__ == '__main__':
    build()
