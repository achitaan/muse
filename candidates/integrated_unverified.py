# okc: problem=muse_bf16_proj
import torch
import triton
import triton.language as tl

@triton.jit
def _gemv_contiguous_k(X, W, Y, N: tl.constexpr, K: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """One program streams several weight rows with contiguous K loads."""
    rows = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    cols = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    for start in range(tl.cdiv(K, BLOCK_K)):
        ks = start * BLOCK_K + cols
        x = tl.load(X + ks, mask=ks < K, other=0).to(tl.float32)
        w = tl.load(W + rows[:, None] * K + ks[None, :], mask=(rows[:, None] < N) & (ks[None, :] < K), other=0).to(tl.float32)
        acc = tl.fma(w, x[None, :], acc)
    values = tl.sum(acc, axis=1)
    tl.store(Y + rows, values, mask=rows < N)

@triton.jit
def _split_k_bf16(X, WT, Partial, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, WEIGHT_N: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, SPLIT_K: tl.constexpr):
    """BF16 tensor-core partial GEMM, with disjoint FP32 scratch writes."""
    pid_n = tl.program_id(0)
    pid_split = tl.program_id(1)
    rows = tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    reduction = tl.arange(0, BLOCK_K)
    partition_k = tl.cdiv(K, SPLIT_K * BLOCK_K) * BLOCK_K
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for step in range(tl.cdiv(partition_k, BLOCK_K)):
        ks = pid_split * partition_k + step * BLOCK_K + reduction
        a = tl.load(X + rows[:, None] * K + ks[None, :], mask=(rows[:, None] < M) & (ks[None, :] < K), other=0)
        b = tl.load(WT + ks[:, None] * WEIGHT_N + cols[None, :], mask=(ks[:, None] < K) & (cols[None, :] < N), other=0)
        acc = tl.dot(a, b, acc, input_precision='ieee')
    dst = Partial + pid_split * M * N + rows[:, None] * N + cols[None, :]
    tl.store(dst, acc, mask=(rows[:, None] < M) & (cols[None, :] < N))

@triton.jit
def _finish_split_k(Partial, Y, ELEMENTS: tl.constexpr, SPLIT_K: tl.constexpr, BLOCK: tl.constexpr):
    indices = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    splits = tl.arange(0, SPLIT_K)
    values = tl.load(Partial + splits[:, None] * ELEMENTS + indices[None, :], mask=indices[None, :] < ELEMENTS, other=0)
    result = tl.sum(values, axis=0)
    tl.store(Y + indices, result, mask=indices < ELEMENTS)

def _small_prepare(weights):
    """Lossless layouts only; no input-dependent work is done here."""
    original = {name: w.contiguous() for (name, w) in weights.items()}
    transposed = {name: w.T.contiguous() for (name, w) in original.items() if w.dtype == torch.bfloat16}
    return {'weights': original, 'weight_t': transposed}

def _small_kernel(data, state):
    x = data['x']
    config = data['config']
    (layer, n) = (config['layer'], config['n'])
    w = state['weights'][layer]
    (m, k) = x.shape
    if w.dtype == torch.float32:
        return torch.mm(x.float(), w[:n].T)
    supported = x.is_cuda and x.dtype == torch.bfloat16 and x.is_contiguous() and (k in (1024, 4096, 8192)) and (k == w.shape[1]) and (0 < n <= w.shape[0]) and (0 < m <= 32)
    if not supported:
        return torch.mm(x, w[:n].T)
    out = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
    if m == 1:
        _gemv_contiguous_k[triton.cdiv(n, 4),](x, w, out, N=n, K=k, BLOCK_N=4, BLOCK_K=1024, num_warps=4)
        return out
    split_k = 16 if n <= 1024 and k >= 4096 else 8
    block_m = 16 if m <= 16 else 32
    partial = torch.empty((split_k, m, n), device=x.device, dtype=torch.float32)
    wt = state['weight_t'][layer]
    _split_k_bf16[triton.cdiv(n, 64), split_k](x, wt, partial, M=m, N=n, K=k, WEIGHT_N=w.shape[0], BLOCK_M=block_m, BLOCK_N=64, BLOCK_K=64, SPLIT_K=split_k, num_warps=4, num_stages=3)
    _finish_split_k[triton.cdiv(m * n, 256),](partial, out, ELEMENTS=m * n, SPLIT_K=split_k, BLOCK=256, num_warps=4)
    return out

@triton.jit
def _large_gemm(X, W, OUT, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, XSM: tl.constexpr, XSK: tl.constexpr, WSN: tl.constexpr, WSK: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, SPLIT_K: tl.constexpr, GROUP_M: tl.constexpr):
    tile = tl.program_id(0)
    split = tl.program_id(1)
    tiles_m = tl.cdiv(M, BM)
    tiles_n = tl.cdiv(N, BN)
    group = tile // (GROUP_M * tiles_n)
    first_m = group * GROUP_M
    group_m = tl.minimum(tiles_m - first_m, GROUP_M)
    in_group = tile % (GROUP_M * tiles_n)
    tile_m = first_m + in_group % group_m
    tile_n = in_group // group_m
    rows = tile_m * BM + tl.arange(0, BM)
    cols = tile_n * BN + tl.arange(0, BN)
    blocks_per_split = tl.cdiv(tl.cdiv(K, BK), SPLIT_K)
    kk = split * blocks_per_split * BK + tl.arange(0, BK)
    x_ptrs = X + rows[:, None] * XSM + kk[None, :] * XSK
    w_ptrs = W + cols[:, None] * WSN + kk[None, :] * WSK
    acc = tl.zeros((BM, BN), tl.float32)
    for step in range(blocks_per_split):
        k_valid = kk < K
        if M % BM == 0 and K % (BK * SPLIT_K) == 0:
            x = tl.load(x_ptrs)
        elif K % (BK * SPLIT_K) == 0:
            x = tl.load(x_ptrs, rows[:, None] < M, 0.0)
        else:
            x = tl.load(x_ptrs, (rows[:, None] < M) & k_valid[None, :], 0.0)
        if N % BN == 0 and K % (BK * SPLIT_K) == 0:
            w = tl.load(w_ptrs)
        else:
            w = tl.load(w_ptrs, (cols[:, None] < N) & k_valid[None, :], 0.0)
        acc = tl.dot(x, tl.trans(w), acc)
        x_ptrs += BK * XSK
        w_ptrs += BK * WSK
        kk += BK
    out_ptrs = OUT + split * M * N + rows[:, None] * N + cols[None, :]
    valid = (rows[:, None] < M) & (cols[None, :] < N)
    if M % BM == 0 and N % BN == 0:
        tl.store(out_ptrs, acc)
    else:
        tl.store(out_ptrs, acc, valid)

@triton.jit
def _large_split_reduce(PARTIALS, OUT, SIZE: tl.constexpr, SPLIT_K: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), tl.float32)
    for split in tl.static_range(SPLIT_K):
        acc += tl.load(PARTIALS + split * SIZE + offs, offs < SIZE, 0.0)
    tl.store(OUT + offs, acc, offs < SIZE)

def _large_config(m, n, k):
    """Static launch choices; pending paired cold-L2 CUDA graph evidence."""
    bm = 32 if m <= 128 else 64
    (bn, bk, split) = (64, 64, 1)
    tiles = triton.cdiv(m, bm) * triton.cdiv(n, bn)
    if m <= 64 and k >= 4096 and (tiles < 96):
        split = 4 if tiles < 48 else 2
    return (bm, bn, bk, split, 4, 3)

def large_bf16_gemm(x, weight, n):
    """Compute x @ weight[:n].T for any positive M and masked tails."""
    (m, k) = x.shape
    (bm, bn, bk, split, warps, stages) = _large_config(m, n, k)
    output = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    if split == 1:
        target = output
    else:
        target = torch.empty((split, m, n), dtype=torch.float32, device=x.device)
    grid = (triton.cdiv(m, bm) * triton.cdiv(n, bn), split)
    _large_gemm[grid](x, weight, target, m, n, k, x.stride(0), x.stride(1), weight.stride(0), weight.stride(1), bm, bn, bk, split, 4, num_warps=warps, num_stages=stages)
    if split != 1:
        _large_split_reduce[triton.cdiv(m * n, 1024),](target, output, m * n, split, 1024, num_warps=4)
    return output

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

def _router_prepare(weights):
    state = {name: w.contiguous() for (name, w) in weights.items()}
    router = state['router']
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    state['rhi'] = hi.T.contiguous()
    state['rlo'] = lo.T.contiguous()
    return state

def router_gemm(x, state, n):
    (m, k) = x.shape
    (bm, bn, bk) = (16 if m <= 16 else 32, 32, 64)
    splits = 16 if m <= 32 else 4
    partial = torch.empty((splits, m, n), device=x.device, dtype=torch.float32)
    out = torch.empty((m, n), device=x.device, dtype=torch.float32)
    _router_products[triton.cdiv(m, bm) * triton.cdiv(n, bn), splits](x, state['rhi'], state['rlo'], partial, m, n, k, bm, bn, bk, splits, num_warps=4, num_stages=2)
    _router_reduce[triton.cdiv(m * n, 256),](partial, out, m * n, splits, 256, num_warps=4)
    return out

def prepare(weights):
    return {"small": _small_prepare(weights), "router": _router_prepare(weights)}


def kernel_fn(data, state):
    x, cfg = data["x"], data["config"]
    n, layer = cfg["n"], cfg["layer"]
    if layer == "router":
        return router_gemm(x, state["router"], n)
    if x.shape[0] <= 32:
        return _small_kernel(data, state["small"])
    w = state["small"]["weights"][layer]
    return large_bf16_gemm(x, w, n)

