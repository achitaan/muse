import triton
import triton.language as tl
@triton.jit
def _pfb(T, W: tl.constexpr, CH: tl.constexpr):
 p = tl.program_id(0).to(tl.int64)
 s = tl.num_programs(0).to(tl.int64) * W
 e = tl.arange(0, W * 32).to(tl.int64)
 b0 = tl.load(T)
 b1 = tl.load(T + 1)
 b2 = tl.load(T + 2)
 b3 = tl.load(T + 3)
 c1 = tl.load(T + 4)
 c2 = tl.load(T + 5)
 c3 = tl.load(T + 6)
 n = tl.load(T + 7)
 for i in range(0, (n + s * 32 - 1) // (s * 32)):
  c = i * s * 32 + p * W * 32 + e
  ok = c < n
  c = tl.minimum(c, n - 1)
  a = tl.where(c < c1, b0 + c * CH, tl.where(c < c2, b1 + (c - c1) * CH, tl.where(c < c3, b2 + (c - c2) * CH, b3 + (c - c3) * CH)))
  a = tl.where(ok, a, 0)
  tl.inline_asm_elementwise('{.reg .pred q; setp.ne.u64 q, $1, 0; @q cp.async.bulk.prefetch.L2.global [$1], 65536;} mov.u32 $0,0;', '=r,l', [a], dtype=tl.int32, is_pure=False, pack=1)

