import sys; sys.path.insert(0, "results/prefill-kv-gather-triton-p7-20261029")
import triton, gk_src
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
c = triton.compile(ASTSource(gk_src._p7_gather_kernel, signature={"KS": "*i64", "VS": "*i64", "IDX": "*i32", "KD": "*i64", "VD": "*i64", "n_rows": "i32", "ROW": "constexpr", "BLOCK": "constexpr"}, constexprs={"ROW": 16, "BLOCK": 64}), target=GPUTarget("cuda", 90, 32), options={"num_warps": 4})
print("compiled sm_90", "shared", c.metadata.shared, "ptx lines", len(c.asm["ptx"].splitlines()))
print([l.strip() for l in c.asm["ptx"].splitlines() if "ld.global" in l][:2])
import collections; print(dict(collections.Counter(t for l in c.asm['ptx'].splitlines() for t in l.split() if t.startswith('ld.global') or t.startswith('st.global'))))
