# okc: problem=muse_bf16_proj
"""Weight-major small-batch GEMM, fused router, and measured cuBLAS fallbacks."""
import torch
import triton
import triton.language as tl


@triton.jit
def _gemv(X, W, Y, N: tl.constexpr, K: tl.constexpr,
          BN: tl.constexpr, BK: tl.constexpr, SPLIT: tl.constexpr):
    rn = tl.program_id(0) * BN + tl.arange(0, BN)
    ss = tl.program_id(1)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BN, BK), tl.float32)
    for step in range(tl.cdiv(K, BK * SPLIT)):
        ks = (ss * tl.cdiv(K, BK * SPLIT) + step) * BK + rk
        a = tl.load(X + ks, ks < K, 0).to(tl.float32)
        w = tl.load(W + rn[:, None] * K + ks[None, :], (rn[:, None] < N) & (ks[None, :] < K), 0).to(tl.float32)
        acc = tl.fma(w, a[None, :], acc)
    tl.store(Y + ss * N + rn, tl.sum(acc, 1), rn < N)


@triton.jit
def _wm_gemm(X, WH, WL, P,
             M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
             BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
             SPLIT: tl.constexpr, DUAL: tl.constexpr):
    rn = tl.program_id(0) * BN + tl.arange(0, BN)
    rm = tl.program_id(1) * BM + tl.arange(0, BM)
    ss = tl.program_id(2)
    rk = tl.arange(0, BK)
    ah = tl.zeros((BN, BM), tl.float32)
    al = tl.zeros((BN, BM), tl.float32)
    steps = tl.cdiv(K, BK * SPLIT)
    for step in range(steps):
        ks = (ss * steps + step) * BK + rk
        w = tl.load(WH + rn[:, None] * K + ks[None, :], (rn[:, None] < N) & (ks[None, :] < K), 0)
        x = tl.load(X + rm[None, :] * K + ks[:, None], (rm[None, :] < M) & (ks[:, None] < K), 0)
        ah = tl.dot(w, x, ah)
        if DUAL:
            lo = tl.load(WL + rn[:, None] * K + ks[None, :], (rn[:, None] < N) & (ks[None, :] < K), 0)
            al = tl.dot(lo, x, al)
    value = ah + al if DUAL else ah
    tl.store(P + ss * M * N + rm[None, :] * N + rn[:, None], value,
             (rn[:, None] < N) & (rm[None, :] < M))


@triton.jit
def _reduce(P, Y, SIZE: tl.constexpr, SPLIT: tl.constexpr, BLOCK: tl.constexpr):
    ii = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    ss = tl.arange(0, SPLIT)
    v = tl.load(P + ss[:, None] * SIZE + ii[None, :], ii[None, :] < SIZE, 0)
    tl.store(Y + ii, tl.sum(v, 0), ii < SIZE)


def prepare(weights):
    state = {name: w.contiguous() for name, w in weights.items()}
    hi = state['router'].to(torch.bfloat16)
    lo = (state['router'] - hi.float()).to(torch.bfloat16)
    state['rhi'], state['rlo'] = hi, lo
    state['rpack'] = torch.cat((hi, lo), dim=0).T.contiguous()
    return state


def kernel_fn(data, state):
    x, cfg = data['x'], data['config']
    m, k = x.shape
    n, layer = cfg['n'], cfg['layer']
    router = layer == 'router'
    if m > 32:
        if router:
            z = torch.mm(x, state['rpack'], out_dtype=torch.float32)
            wn = state['router'].shape[0]
            return z[:, :n] + z[:, wn:wn+n]
        return torch.mm(x, state[layer][:n].T)
    out = torch.empty((m, n), device=x.device, dtype=torch.float32 if router else torch.bfloat16)
    if m == 1:
        split = 4 if router else 1
        p = torch.empty((split, n), device=x.device, dtype=torch.float32) if split > 1 else out
        _gemv[(triton.cdiv(n, 4), split)](x, state[layer], p, n, k, 4, 1024, split, num_warps=4)
        if split > 1:
            _reduce[(triton.cdiv(n, 256),)](p, out, n, split, 256)
        return out
    if not router and layer != 'pooler':
        return torch.mm(x, state[layer][:n].T)
    bm = 16 if m <= 16 else 32
    bn, bk = (16, 128) if router else (32, 128)
    split = 16 if router else 8
    p = torch.empty((split, m, n), device=x.device, dtype=torch.float32)
    wh = state['rhi'] if router else state[layer]
    wl = state['rlo'] if router else wh
    _wm_gemm[(triton.cdiv(n, bn), triton.cdiv(m, bm), split)](
        x, wh, wl, p, m, n, k, bm, bn, bk, split, router, num_warps=4, num_stages=2)
    _reduce[(triton.cdiv(m*n, 256),)](p, out, m*n, split, 256, num_warps=4)
    return out
