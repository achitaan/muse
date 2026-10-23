import triton
import triton.language as tl

@triton.jit
def _router_products(X, HI, LO, P, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, SPLITS: tl.constexpr):
    im = tl.program_id(0) // tl.cdiv(N, BN)
    jn = tl.program_id(0) % tl.cdiv(N, BN)
    split = tl.program_id(1)
    rows = im * BM + tl.arange(0, BM)
    cols = jn * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    steps = tl.cdiv(K, BK * SPLITS)
    ah = tl.zeros((BM, BN), tl.float32)
    al = tl.zeros((BM, BN), tl.float32)
    for step in range(steps):
        ks = (split * steps + step) * BK + kk
        a = tl.load(X + rows[:, None] * K + ks[None, :], (rows[:, None] < M) & (ks[None, :] < K), 0)
        hi = tl.load(HI + ks[:, None] * N + cols[None, :], (ks[:, None] < K) & (cols[None, :] < N), 0)
        lo = tl.load(LO + ks[:, None] * N + cols[None, :], (ks[:, None] < K) & (cols[None, :] < N), 0)
        ah = tl.dot(a, hi, ah)
        al = tl.dot(a, lo, al)
    offs = split * M * N + rows[:, None] * N + cols[None, :]
    tl.store(P + offs, ah + al, (rows[:, None] < M) & (cols[None, :] < N))

@triton.jit
def _router_reduce(P, Y, SIZE: tl.constexpr, SPLITS: tl.constexpr, BLOCK: tl.constexpr):
    ids = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    ss = tl.arange(0, SPLITS)
    vals = tl.load(P + ss[:, None] * SIZE + ids[None, :], ids[None, :] < SIZE, 0)
    tl.store(Y + ids, tl.sum(vals, 0), ids < SIZE)

@triton.jit
def _router_gemv(X, W, P, N: tl.constexpr, K: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, SPLITS: tl.constexpr):
    rows = tl.program_id(0) * BN + tl.arange(0, BN)
    split = tl.program_id(1)
    lanes = tl.arange(0, BK)
    steps = tl.cdiv(K, BK * SPLITS)
    acc = tl.zeros((BN, BK), tl.float32)
    for step in range(steps):
        kk = (split * steps + step) * BK + lanes
        x = tl.load(X + kk, kk < K, 0).to(tl.float32)
        w = tl.load(W + rows[:, None] * K + kk[None, :], (rows[:, None] < N) & (kk[None, :] < K), 0)
        acc = tl.fma(w, x[None, :], acc)
    tl.store(P + split * N + rows, tl.sum(acc, 1), rows < N)
