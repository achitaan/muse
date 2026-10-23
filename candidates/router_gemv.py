# okc: problem=muse_bf16_proj
"""Fuse the allowed BF16 hi/lo router products with FP32 partial sums."""
import torch
import triton
import triton.language as tl


@triton.jit
def _router_products(X, HI, LO, P, M: tl.constexpr, N: tl.constexpr,
                     K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                     BK: tl.constexpr, SPLITS: tl.constexpr):
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
        a = tl.load(X + rows[:, None] * K + ks[None, :],
                    (rows[:, None] < M) & (ks[None, :] < K), 0)
        hi = tl.load(HI + ks[:, None] * N + cols[None, :],
                     (ks[:, None] < K) & (cols[None, :] < N), 0)
        lo = tl.load(LO + ks[:, None] * N + cols[None, :],
                     (ks[:, None] < K) & (cols[None, :] < N), 0)
        ah = tl.dot(a, hi, ah)
        al = tl.dot(a, lo, al)
    offs = split * M * N + rows[:, None] * N + cols[None, :]
    tl.store(P + offs, ah + al, (rows[:, None] < M) & (cols[None, :] < N))


@triton.jit
def _router_reduce(P, Y, SIZE: tl.constexpr, SPLITS: tl.constexpr,
                   BLOCK: tl.constexpr):
    ids = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    ss = tl.arange(0, SPLITS)
    vals = tl.load(P + ss[:, None] * SIZE + ids[None, :], ids[None, :] < SIZE, 0)
    tl.store(Y + ids, tl.sum(vals, 0), ids < SIZE)


def prepare(weights):
    state = {name: w.contiguous() for name, w in weights.items()}
    router = state["router"]
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    state["rhi"] = hi.T.contiguous()
    state["rlo"] = lo.T.contiguous()
    return state


def router_gemm(x, state, n):
    m, k = x.shape
    # BF16 input values remain exact. The weight decomposition is explicitly
    # allowed for this FP32 router at all batch sizes by the task specification.
    bm, bn, bk = (16 if m <= 16 else 32), 32, 64
    splits = 16 if m <= 32 else 4
    partial = torch.empty((splits, m, n), device=x.device, dtype=torch.float32)
    out = torch.empty((m, n), device=x.device, dtype=torch.float32)
    _router_products[(triton.cdiv(m, bm) * triton.cdiv(n, bn), splits)](
        x, state["rhi"], state["rlo"], partial, m, n, k,
        bm, bn, bk, splits, num_warps=4, num_stages=2)
    _router_reduce[(triton.cdiv(m*n, 256),)](
        partial, out, m*n, splits, 256, num_warps=4)
    return out


@triton.jit
def _router_gemv(X, W, P, N: tl.constexpr, K: tl.constexpr,
                 BN: tl.constexpr, BK: tl.constexpr, SPLITS: tl.constexpr):
    rows = tl.program_id(0) * BN + tl.arange(0, BN)
    split = tl.program_id(1)
    lanes = tl.arange(0, BK)
    steps = tl.cdiv(K, BK * SPLITS)
    acc = tl.zeros((BN, BK), tl.float32)
    for step in range(steps):
        kk = (split * steps + step) * BK + lanes
        x = tl.load(X + kk, kk < K, 0).to(tl.float32)
        w = tl.load(W + rows[:, None] * K + kk[None, :],
                    (rows[:, None] < N) & (kk[None, :] < K), 0)
        acc = tl.fma(w, x[None, :], acc)
    tl.store(P + split * N + rows, tl.sum(acc, 1), rows < N)


def kernel_fn(data, state):
    x, cfg = data["x"], data["config"]
    if cfg["layer"] == "router":
        if x.shape[0] == 1:
            n, k, splits = cfg["n"], x.shape[1], 4
            partial = torch.empty((splits, n), device=x.device, dtype=torch.float32)
            out = torch.empty((1, n), device=x.device, dtype=torch.float32)
            _router_gemv[(triton.cdiv(n, 4), splits)](
                x, state["router"], partial, n, k, 4, 512, splits, num_warps=4)
            _router_reduce[(triton.cdiv(n, 256),)](
                partial, out, n, splits, 256, num_warps=4)
            return out
        return router_gemm(x, state, cfg["n"])
    return torch.mm(x, state[cfg["layer"]][:cfg["n"]].T)
