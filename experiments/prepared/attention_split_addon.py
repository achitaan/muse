
# Local full-attention experiment: exact partitioned online softmax.
# Every visible key belongs to exactly one partition; no score-based skipping.
@triton.jit
def _achita_attention_parts(Q, KC, VC, TABLE, SEQ, KS, VS, PO, PL,
                            HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr,
                            L: tl.constexpr, G: tl.constexpr, PAGE: tl.constexpr,
                            TS: tl.constexpr, WINDOW: tl.constexpr,
                            SM: tl.constexpr, PART: tl.constexpr,
                            SPLITS: tl.constexpr, BM: tl.constexpr,
                            BN: tl.constexpr):
    bh = tl.program_id(0)
    part = tl.program_id(1)
    batch, head = bh // HKV, bh % HKV
    rr = tl.arange(0, BM)
    dd = tl.arange(0, D)
    nn = tl.arange(0, BN)
    qi, qh = rr // G, head * G + rr % G
    seq = tl.load(SEQ + batch)
    q = tl.load(Q + ((batch * L + qi[:, None]) * HQ + qh[:, None]) * D + dd[None, :],
                mask=qi[:, None] < L, other=0).to(tl.bfloat16)
    base = 0
    if WINDOW:
        base = tl.maximum(0, seq + 1 - WINDOW)
    start = base + part * PART
    m = tl.full((BM,), -1.0e20, tl.float32)
    den = tl.zeros((BM,), tl.float32)
    acc = tl.zeros((BM, D), tl.float32)
    scale = SM * tl.load(KS) * 1.4426950408889634
    if start < seq + L:
        for block in range(tl.cdiv(PART, BN)):
            pos = start + block * BN + nn
            valid = (pos < seq + L) & (pos < start + PART)
            physical = tl.load(TABLE + batch * TS + pos // PAGE, mask=valid, other=0)
            row = physical * PAGE + pos % PAGE
            offset = (row[None, :] * HKV + head) * D + dd[:, None]
            k = tl.load(KC + offset, mask=valid[None, :], other=0).to(tl.bfloat16)
            score = tl.dot(q, k) * scale
            visible = valid[None, :] & (pos[None, :] <= seq + qi[:, None]) & (qi[:, None] < L)
            if WINDOW:
                visible &= pos[None, :] > seq + qi[:, None] - WINDOW
            score = tl.where(visible, score, -1.0e20)
            new_m = tl.maximum(m, tl.max(score, 1))
            correction = tl.exp2(m - new_m)
            probability = tl.where(visible, tl.exp2(score - new_m[:, None]), 0.0)
            v = tl.load(VC + tl.trans(offset), mask=valid[:, None], other=0).to(tl.bfloat16)
            acc = acc * correction[:, None]
            acc = tl.dot(probability.to(tl.bfloat16), v, acc)
            den = den * correction + tl.sum(probability, 1)
            m = new_m
    # Keep FP32 partition results and log normalizers for stable reduction.
    normalized = acc / tl.maximum(den[:, None], 1.0e-30) * tl.load(VS)
    po = ((bh * SPLITS + part) * BM + rr[:, None]) * D + dd[None, :]
    tl.store(PO + po, normalized)
    tl.store(PL + (bh * SPLITS + part) * BM + rr,
             tl.where(den > 0, m + tl.log2(den), -float('inf')))


@triton.jit
def _achita_attention_merge(PO, PL, QKV, OUT, HQ: tl.constexpr, HKV: tl.constexpr,
                            L: tl.constexpr, G: tl.constexpr, D: tl.constexpr,
                            SPLITS: tl.constexpr, BM: tl.constexpr,
                            BS: tl.constexpr, STRIDE: tl.constexpr):
    bh = tl.program_id(0)
    rr = tl.program_id(1)
    parts = tl.arange(0, BS)
    dd = tl.arange(0, D)
    lse = tl.load(PL + (bh * SPLITS + parts) * BM + rr, mask=parts < SPLITS, other=-float('inf'))
    weights = tl.exp2(lse - tl.max(lse, 0))
    weights = weights / tl.sum(weights, 0)
    values = tl.load(PO + ((bh * SPLITS + parts[:, None]) * BM + rr) * D + dd[None, :],
                     mask=parts[:, None] < SPLITS, other=0)
    result = tl.sum(values * weights[:, None], 0)
    batch, head = bh // HKV, bh % HKV
    token, qh = rr // G, head * G + rr % G
    # The reference rounds attention output to BF16 before the output RMSNorm.
    result = result.to(tl.bfloat16).to(tl.float32)
    result *= tl.rsqrt(tl.sum(result * result, 0) / D + 1.0e-5)
    gate = tl.load(QKV + (batch * L + token) * STRIDE + (HQ + 2 * HKV + qh) * D + dd).to(tl.float32)
    result *= 1.0 / (1.0 + tl.exp(-gate))
    tl.store(OUT + ((batch * L + token) * HQ + qh) * D + dd, result)


def kernel_fn(data):
    cfg = data['config']
    b, length = cfg['batch'], cfg['q_len']
    hq, hkv, d = cfg['num_q_heads'], cfg['num_kv_heads'], cfg['head_dim']
    tokens = b * length
    qkv = data['qkv'].view(tokens, -1)
    q = torch.empty((tokens, hq, d), device=qkv.device, dtype=torch.float8_e4m3fn)
    _achita_qkv_preprocess[(tokens, hq + hkv)](
        qkv, data['positions'], data['cache_loc'], data['k_cache'], data['v_cache'],
        data['k_scale'], data['v_scale'], q, hq, hkv, d, qkv.stride(0),
        math.log(cfg['rope_theta']) if cfg['rope_theta'] else 0.0,
        bool(cfg['rope_theta']), num_warps=4)
    group = hq // hkv
    bm = max(16, triton.next_power_of_2(length * group))
    part = 1024 if cfg['window'] else 2048
    capacity = min(data['page_table'].shape[1] * cfg['page_size'],
                   cfg['window'] + length - 1) if cfg['window'] else data['page_table'].shape[1] * cfg['page_size']
    splits = triton.cdiv(capacity, part)
    partial = torch.empty((b * hkv, splits, bm, d), device=q.device, dtype=torch.float32)
    lse = torch.empty((b * hkv, splits, bm), device=q.device, dtype=torch.float32)
    out = torch.empty((tokens, hq, d), device=q.device, dtype=torch.bfloat16)
    _achita_attention_parts[(b * hkv, splits)](
        q, data['k_cache'], data['v_cache'], data['page_table'], data['seq_lens'],
        data['k_scale'], data['v_scale'], partial, lse, hq, hkv, d, length,
        group, cfg['page_size'], data['page_table'].shape[1], cfg['window'],
        cfg['sm_scale'] * cfg['qk_scale_factor'], part, splits, bm, 64,
        num_warps=4, num_stages=2)
    _achita_attention_merge[(b * hkv, length * group)](
        partial, lse, qkv, out, hq, hkv, length, group, d, splits, bm,
        triton.next_power_of_2(splits), qkv.stride(0),
        num_warps=8 if splits > 128 else 4)
    return out.view(b, length, hq * d)
