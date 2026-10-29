"""Build decoder_fp8pv.py: D9 with the retained standalone Attention CUDA (FP8 P·V for one-query attention).

Python launch/plan code in D9 and best/attention_checked.py is identical apart from docstrings and the
`_native_attn_` name prefix (checked in this session), so only the CUDA constant and extension name change.
"""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def const(src, name):
    t = ast.parse(src)
    n = next(n for n in t.body if isinstance(n, ast.Assign) and getattr(n.targets[0], 'id', '') == name)
    seg = ast.get_source_segment(src, n.value)
    assert seg == repr(n.value.value) and src.count(seg) == 1
    return seg, n.value.value


def main():
    parent = open(ROOT / 'best/decoder_verified.py', encoding='utf-8', newline='').read()
    attn = open(ROOT / 'best/attention_checked.py', encoding='utf-8', newline='').read()
    seg_old, old_cuda = const(parent, '_native_attn_CUDA_SRC')
    _, new_cuda = const(attn, 'CUDA_SRC')
    out = parent.replace(seg_old, repr(new_cuda))
    assert out.count("okc_attn_v298{tag}_90a") == 1
    out = out.replace("okc_attn_v298{tag}_90a", "okc_attn_v298f8pv{tag}_90a")
    ast.parse(out)
    p = HERE / 'decoder_fp8pv.py'
    p.write_text(out, encoding='utf-8', newline='')
    info = {'parent_sha256': hashlib.sha256(parent.encode()).hexdigest(),
            'attention_source_sha256': hashlib.sha256(attn.encode()).hexdigest(),
            'candidate_sha256': hashlib.sha256(out.encode()).hexdigest(), 'bytes': len(out.encode()),
            'old_cuda_sha256': hashlib.sha256(old_cuda.encode()).hexdigest(),
            'new_cuda_sha256': hashlib.sha256(new_cuda.encode()).hexdigest()}
    print(json.dumps(info, indent=2))


if __name__ == '__main__':
    main()
