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
