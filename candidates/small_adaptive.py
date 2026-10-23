# okc: problem=muse_bf16_proj
"""Adaptive small-M BF16 candidate; all reductions accumulate in FP32.

This file is standalone so its prepare/kernel_fn can be checked directly. The
router deliberately uses the original FP32 weight through PyTorch. Larger and
unsupported BF16 inputs also retain a PyTorch fallback.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _gemv_contiguous_k(
    X, W, Y,
    N: tl.constexpr, K: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """One program streams several weight rows with contiguous K loads."""
    rows = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    cols = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    for start in range(tl.cdiv(K, BLOCK_K)):
        ks = start * BLOCK_K + cols
        x = tl.load(X + ks, mask=ks < K, other=0).to(tl.float32)
        w = tl.load(
            W + rows[:, None] * K + ks[None, :],
            mask=(rows[:, None] < N) & (ks[None, :] < K),
            other=0,
        ).to(tl.float32)
        acc = tl.fma(w, x[None, :], acc)
    values = tl.sum(acc, axis=1)
    tl.store(Y + rows, values, mask=rows < N)


@triton.jit
def _split_k_bf16(
    X, WT, Partial,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    WEIGHT_N: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    """FP32-accumulated GEMM: BF16 direct output, or FP32 split partials."""
    pid_n = tl.program_id(0)
    pid_split = tl.program_id(1)
    rows = tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    reduction = tl.arange(0, BLOCK_K)
    # Supported shapes have K divisible by SPLIT_K * BLOCK_K. Keeping the
    # partition contiguous minimizes pointer arithmetic and preserves coalescing.
    partition_k = tl.cdiv(K, SPLIT_K * BLOCK_K) * BLOCK_K
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for step in range(tl.cdiv(partition_k, BLOCK_K)):
        ks = pid_split * partition_k + step * BLOCK_K + reduction
        a = tl.load(
            X + rows[:, None] * K + ks[None, :],
            mask=(rows[:, None] < M) & (ks[None, :] < K),
            other=0,
        )
        b = tl.load(
            WT + ks[:, None] * WEIGHT_N + cols[None, :],
            mask=(ks[:, None] < K) & (cols[None, :] < N),
            other=0,
        )
        acc = tl.dot(a, b, acc, input_precision="ieee")
    dst = Partial + pid_split * M * N + rows[:, None] * N + cols[None, :]
    tl.store(dst, acc, mask=(rows[:, None] < M) & (cols[None, :] < N))


@triton.jit
def _finish_split_k(
    Partial, Y,
    ELEMENTS: tl.constexpr, SPLIT_K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    indices = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    splits = tl.arange(0, SPLIT_K)
    values = tl.load(
        Partial + splits[:, None] * ELEMENTS + indices[None, :],
        mask=indices[None, :] < ELEMENTS,
        other=0,
    )
    result = tl.sum(values, axis=0)
    # Y is BF16: one final conversion, after every FP32 partial has been summed.
    tl.store(Y + indices, result, mask=indices < ELEMENTS)


def prepare(weights):
    """Lossless layouts only; no input-dependent work is done here."""
    original = {name: w.contiguous() for name, w in weights.items()}
    transposed = {
        name: w.T.contiguous()
        for name, w in original.items()
        if w.dtype == torch.bfloat16
    }
    return {"weights": original, "weight_t": transposed}


def _adaptive_split_k(n):
    """Aim for about 128 CTAs, using the smallest power-of-two split <= 8."""
    n_tiles = triton.cdiv(n, 64)
    if n_tiles >= 128:
        return 1
    if n_tiles >= 64:
        return 2
    if n_tiles >= 32:
        return 4
    return 8


def kernel_fn(data, state):
    x = data["x"]
    config = data["config"]
    layer, n = config["layer"], config["n"]
    w = state["weights"][layer]
    m, k = x.shape
    if w.dtype == torch.float32:
        return torch.mm(x.float(), w[:n].T)

    supported = (
        x.is_cuda and x.dtype == torch.bfloat16 and x.is_contiguous()
        and k in (1024, 4096, 8192) and k == w.shape[1]
        and 0 < n <= w.shape[0] and 0 < m <= 32
    )
    if not supported:
        return torch.mm(x, w[:n].T)

    out = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
    if m == 1:
        _gemv_contiguous_k[(triton.cdiv(n, 4),)](
            x, w, out, N=n, K=k, BLOCK_N=4, BLOCK_K=1024,
            num_warps=4,
        )
        return out

    # A 128-tile N grid already nearly covers H200's 132 SMs. Additional K
    # partitions there add scratch traffic and a reduction launch. Smaller N
    # grids use only enough partitions to expose roughly 128 independent CTAs.
    split_k = _adaptive_split_k(n)
    block_m = 16 if m <= 16 else 32
    if split_k == 1:
        # The dot accumulator is FP32; this store performs the final BF16 cast.
        target = out
    else:
        target = torch.empty((split_k, m, n), device=x.device, dtype=torch.float32)
    wt = state["weight_t"][layer]
    _split_k_bf16[(triton.cdiv(n, 64), split_k)](
        x, wt, target,
        M=m, N=n, K=k, WEIGHT_N=w.shape[0],
        BLOCK_M=block_m, BLOCK_N=64, BLOCK_K=64, SPLIT_K=split_k,
        num_warps=4, num_stages=3,
    )
    if split_k != 1:
        _finish_split_k[(triton.cdiv(m * n, 256),)](
            target, out, ELEMENTS=m * n, SPLIT_K=split_k, BLOCK=256,
            num_warps=4,
        )
    return out
