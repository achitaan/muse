# okc: problem=muse_bf16_proj
"""Conventional BF16 GEMM candidate for M >= 32, with bounded split K.

BF16 weights and activations are loaded at their given precision, products
accumulate in FP32, and the final result is rounded to BF16 once. The split-K
path writes FP32 partials and sums those partials in a separate kernel.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _large_gemm(
    X, W, OUT,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    XSM: tl.constexpr, XSK: tl.constexpr,
    WSN: tl.constexpr, WSK: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    SPLIT_K: tl.constexpr, GROUP_M: tl.constexpr,
):
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
    # All blocks in one split own an adjacent, BK-aligned part of K.
    blocks_per_split = tl.cdiv(tl.cdiv(K, BK), SPLIT_K)
    kk = split * blocks_per_split * BK + tl.arange(0, BK)
    x_ptrs = X + rows[:, None] * XSM + kk[None, :] * XSK
    # W remains row-major [N,K]. Loading [BN,BK] keeps K contiguous;
    # its transpose is the second [BK,BN] operand to the matrix product.
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
    # OUT is BF16 with SPLIT_K=1 and FP32 otherwise. No BF16 partial sums.
    if M % BM == 0 and N % BN == 0:
        tl.store(out_ptrs, acc)
    else:
        tl.store(out_ptrs, acc, valid)


@triton.jit
def _large_split_reduce(
    PARTIALS, OUT,
    SIZE: tl.constexpr, SPLIT_K: tl.constexpr, BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), tl.float32)
    for split in tl.static_range(SPLIT_K):
        acc += tl.load(PARTIALS + split * SIZE + offs, offs < SIZE, 0.0)
    tl.store(OUT + offs, acc, offs < SIZE)


def _large_config(m, n, k):
    """Static launch choices; pending paired cold-L2 CUDA graph evidence."""
    bm = 32 if m <= 128 else 64
    bn, bk, split = 64, 64, 1
    tiles = triton.cdiv(m, bm) * triton.cdiv(n, bn)
    if m <= 64 and k >= 4096 and tiles < 96:
        # For M=32: QKV has 36 CTAs -> 144; pooler has 64 -> 128.
        # Full-width unpooler already has 128 CTAs and avoids a second launch.
        split = 4 if tiles < 48 else 2
    return bm, bn, bk, split, 4, 3


def prepare(weights):
    """Keep only lossless contiguous copies of weights; no input-derived state."""
    return {name: weight.contiguous() for name, weight in weights.items()}


def large_bf16_gemm(x, weight, n):
    """Compute x @ weight[:n].T for any positive M and masked tails."""
    m, k = x.shape
    bm, bn, bk, split, warps, stages = _large_config(m, n, k)
    output = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    if split == 1:
        target = output
    else:
        target = torch.empty((split, m, n), dtype=torch.float32, device=x.device)
    grid = (triton.cdiv(m, bm) * triton.cdiv(n, bn), split)
    _large_gemm[grid](
        x, weight, target, m, n, k,
        x.stride(0), x.stride(1), weight.stride(0), weight.stride(1),
        bm, bn, bk, split, 4, num_warps=warps, num_stages=stages,
    )
    if split != 1:
        _large_split_reduce[(triton.cdiv(m * n, 1024),)](
            target, output, m * n, split, 1024, num_warps=4,
        )
    return output


def kernel_fn(data, state):
    x, config = data["x"], data["config"]
    layer, n = config["layer"], config["n"]
    weight = state[layer]
    # Standalone correctness fallbacks. The parent integrates dedicated kernels
    # for router and small M into its final candidate.
    if layer == "router":
        return torch.mm(x.float(), weight[:n].t())
    if x.shape[0] < 32:
        return torch.mm(x, weight[:n].t())
    return large_bf16_gemm(x, weight, n)
