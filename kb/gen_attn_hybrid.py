# achitaan: generates kb/v1/attn_hybrid.py (run from the kbase worktree root)
import ast, base64, zlib

K = open('kb/muse_attention_decode.py', encoding='utf-8').read()
A = open('../alygg/kernels/muse_attention_decode.py', encoding='utf-8').read()
OLD = 'name=f"okc_attn_v298{tag}_90a"'
assert A.count(OLD) == 1, A.count(OLD)
A = A.replace(OLD, 'name=f"aly_okc_attn_v298{tag}_90a"')  # achitaan: unique extension name in a shared process


def enc(s):
    return base64.b85encode(zlib.compress(s.encode(), 9)).decode()


TMPL = '''# okc: problem=muse_attention_decode
# achitaan hybrid attention: each scored shape runs the faster of two public Apache-2.0 files, keell0renz
# (github.com/keell0renz/muse-open-kernel-challenge) and alygg77 (github.com/alygg77/okc-muse-spark-kernels; its
# extension name prefixed "aly_", otherwise unmodified). alygg77 serves the sliding-window shapes where it beat
# keell0renz by 3.7-5.2 percent on the same judge (checks 6ae275da / e54231af): decode 1, decode 16, verify 16;
# keell0renz serves everything else. Full license / notice texts are inside the two embedded sources.
import zlib as _z, base64 as _b, types as _t
_K = @K@
_A = @A@


def _load(name, blob):
    m = _t.ModuleType(name)
    m.__file__ = name + ".py"
    exec(compile(_z.decompress(_b.b85decode(blob)).decode(), name + ".py", "exec"), m.__dict__)
    return m


_MK = _load("kb_attn", _K)
_MA = _load("aly_attn", _A)
_ALY = {(1, 1), (1, 16), (8, 16)}  # (q_len, batch), sliding layers only


def kernel_fn(data):
    cfg = data["config"]
    if cfg["window"] and (cfg["q_len"], cfg["batch"]) in _ALY:
        return _MA.kernel_fn(data)
    return _MK.kernel_fn(data)
'''

pre, rest = TMPL.split('@K@')
mid, post = rest.split('@A@')
src = pre + repr(enc(K)) + mid + repr(enc(A)) + post
ast.parse(src)
open('kb/v1/attn_hybrid.py', 'w', encoding='utf-8', newline='\n').write(src)
print(len(src.encode()))
