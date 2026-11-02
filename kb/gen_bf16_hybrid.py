# achitaan: generates kb/v1/bf16_hybrid.py (run from the kbase worktree root)
import ast, base64, zlib

K = open('kb/muse_bf16_proj.py', encoding='utf-8').read()
A = open('../alygg/kernels/muse_bf16_proj.py', encoding='utf-8').read()


def enc(s):
    return base64.b85encode(zlib.compress(s.encode(), 9)).decode()


TMPL = '''# okc: problem=muse_bf16_proj
# achitaan hybrid dense projections: each scored (layer, m) cell runs the faster of two public Apache-2.0 files
# (keell0renz github.com/keell0renz/muse-open-kernel-challenge, and alygg77 github.com/alygg77/okc-muse-spark-kernels),
# both unmodified; alygg77 for the cells where it beat keell0renz by more than 3 percent on the same judge (checks
# 70d8df28 / c3673f3d), keell0renz everywhere else. Each module JIT-builds into its own extension directory so
# same-named extensions never mix. Full license / notice texts are inside the two embedded sources.
import os as _os, zlib as _z, base64 as _b, types as _t
_K = @K@
_A = @A@
_BASE = _os.environ.get("TORCH_EXTENSIONS_DIR") or _os.path.join(_os.path.expanduser("~"), ".cache", "torch_extensions")


class _Dir:
    def __init__(self, sub):
        self.d = _os.path.join(_BASE, sub)

    def __enter__(self):
        self.old = _os.environ.get("TORCH_EXTENSIONS_DIR")
        _os.environ["TORCH_EXTENSIONS_DIR"] = self.d

    def __exit__(self, *a):
        if self.old is None:
            _os.environ.pop("TORCH_EXTENSIONS_DIR", None)
        else:
            _os.environ["TORCH_EXTENSIONS_DIR"] = self.old


def _load(name, blob):
    m = _t.ModuleType(name)
    m.__file__ = name + ".py"
    with _Dir(name):
        exec(compile(_z.decompress(_b.b85decode(blob)).decode(), name + ".py", "exec"), m.__dict__)
    return m


_MK = _load("kb_bf16", _K)
_MA = _load("aly_bf16", _A)
ALY = {("qkv_gate", 1), ("qkv_gate", 16), ("pooler", 8), ("router", 8), ("unpooler", 8)}


def prepare(weights):
    with _Dir("kb_bf16"):
        sk = _MK.prepare(weights)
    with _Dir("aly_bf16"):
        sa = _MA.prepare(weights)
    return {"k": sk, "a": sa}


def kernel_fn(data, state):
    cfg = data["config"]
    if (cfg["layer"], cfg["m"]) in ALY:
        return _MA.kernel_fn(data, state["a"])
    return _MK.kernel_fn(data, state["k"])
'''

# b85 text can contain '@': split on the placeholders once instead of chained str.replace
pre, rest = TMPL.split('@K@')
mid, post = rest.split('@A@')
src = pre + repr(enc(K)) + mid + repr(enc(A)) + post
ast.parse(src)
open('kb/v1/bf16_hybrid.py', 'w', encoding='utf-8', newline='\n').write(src)
print(len(src.encode()))
