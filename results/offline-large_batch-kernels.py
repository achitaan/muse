import triton
import triton.language as tl

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
