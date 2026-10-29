import triton
import triton.language as tl
@triton.jit
def _pfk(T, W: tl.constexpr):
 p = tl.program_id(0).to(tl.int64)
 s = tl.num_programs(0).to(tl.int64) * W
 e = tl.arange(0, W * 32)
 l = (e % 32).to(tl.int64)
 w = (e // 32).to(tl.int64)
 b0 = tl.load(T)
 b1 = tl.load(T + 1)
 b2 = tl.load(T + 2)
 b3 = tl.load(T + 3)
 c1 = tl.load(T + 4)
 c2 = tl.load(T + 5)
 c3 = tl.load(T + 6)
 n = tl.load(T + 7)
 for i in range(0, (n + s - 1) // s):
  c = tl.minimum(i * s + p * W + w, n - 1)
  a = tl.where(c < c1, b0 + c * 4096, tl.where(c < c2, b1 + (c - c1) * 4096, tl.where(c < c3, b2 + (c - c2) * 4096, b3 + (c - c3) * 4096)))
  tl.inline_asm_elementwise('prefetch.global.L2 [$1];prefetch.global.L2 [$1+1024];prefetch.global.L2 [$1+2048];prefetch.global.L2 [$1+3072];mov.u32 $0,0;', '=r,l', [a + l * 32], dtype=tl.int32, is_pure=False, pack=1)
