"""Compare the exact routing distributions of muse_moe_w4a16 vs muse_decode_layer on CPU.

Directly uses the mathematical definitions from muse_moe_w4a16/reference.py and
muse_decode_layer/reference.py (no GPU required).
"""
import math
import torch

N_EXPERTS = 256
LOCAL_EXPERTS = 32
TOP_K = 8
HIDDEN = 8192
POPULARITY = 0.03
STREAM_SEED = 5354
STREAM_MEAN = 0.7
STREAM_SHARED, STREAM_RANK = 0.9, 4
ROUTER_STD, ROUTER_ALIGN, ROUTER_BIAS_STD = 0.045, 0.07, 0.055
DRAFT_SHARE = 0.6
ROUTER_MULTIPLIER = 0.1767766952966369
RMS_EPS = 1e-5


def moe_track_sample(bs: int, seed: int):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    scores = torch.rand((bs, N_EXPERTS), generator=gen)
    z = torch.randn((N_EXPERTS,), generator=gen)
    u = torch.rand((N_EXPERTS,), generator=gen).clamp(min=1e-12)
    scores = scores + POPULARITY * z / (-u.log()).sqrt()
    ids = scores.topk(TOP_K, dim=-1).indices
    pairs = torch.bincount((ids // LOCAL_EXPERTS).flatten(), minlength=N_EXPERTS // LOCAL_EXPERTS)
    ids = (ids - LOCAL_EXPERTS * int(pairs.argmax())) % N_EXPERTS
    local = ids[ids < LOCAL_EXPERTS]
    counts = torch.bincount(local, minlength=LOCAL_EXPERTS) if local.numel() else torch.zeros(LOCAL_EXPERTS, dtype=torch.long)
    return int(local.numel()), int((counts > 0).sum()), counts


def decode_track_sample(batch: int, q_len: int, seed: int, w_seed: int = 5353):
    tokens = batch * q_len
    s_gen = torch.Generator().manual_seed(STREAM_SEED)
    mean = torch.randn((HIDDEN,), generator=s_gen)
    shared = torch.randn((STREAM_RANK, HIDDEN), generator=s_gen)

    r_gen = torch.Generator().manual_seed(w_seed + seed * 17)
    w = ROUTER_STD * torch.randn((N_EXPERTS, HIDDEN), generator=r_gen)
    align = ROUTER_ALIGN * torch.randn((N_EXPERTS, 1), generator=r_gen)
    router = w + align * mean[None, :] / math.sqrt(HIDDEN)
    bias = ROUTER_BIAS_STD * torch.randn((N_EXPERTS,), generator=r_gen)
    pre_ffn = 0.1 * torch.randn((HIDDEN,), generator=r_gen)

    gen = torch.Generator().manual_seed(seed)
    coef = torch.randn((tokens, STREAM_RANK), generator=gen)
    scale = torch.ones((batch, q_len, 1))
    if 1 < q_len <= 8:
        scale[:, 1:] = DRAFT_SHARE
    scale = scale.reshape(tokens, 1)
    stream = (scale * (STREAM_MEAN * mean + math.sqrt(STREAM_SHARED / STREAM_RANK) * coef @ shared)
              + (1 - STREAM_SHARED * scale * scale).sqrt() * torch.randn((tokens, HIDDEN), generator=gen))
    # In decode_layer, stream1 = alpha1 * stream + beta1 * rms(a); mean/shared structure is dominated by stream
    h = stream * torch.rsqrt(stream.pow(2).mean(-1, keepdim=True) + RMS_EPS) * (1.0 + pre_ffn)
    scores = torch.sigmoid((h @ router.T) * ROUTER_MULTIPLIER)
    biased = scores + bias
    ids = biased.topk(TOP_K, dim=-1).indices
    pairs = torch.bincount((ids // LOCAL_EXPERTS).flatten(), minlength=N_EXPERTS // LOCAL_EXPERTS)
    ids = (ids - LOCAL_EXPERTS * int(pairs.argmax())) % N_EXPERTS
    local = ids[ids < LOCAL_EXPERTS]
    counts = torch.bincount(local, minlength=LOCAL_EXPERTS) if local.numel() else torch.zeros(LOCAL_EXPERTS, dtype=torch.long)
    return int(local.numel()), int((counts > 0).sum()), counts


def summarize():
    shapes = [("decode 1", 1, 1), ("verify 1", 1, 8), ("decode 16", 16, 1),
              ("decode 32", 32, 1), ("verify 16", 16, 8), ("verify 32", 32, 8)]
    mb_per_exp = (8192 * 2048 + 8192 * 256 + 4096 * 2048 + 4096 * 256) / 1e6  # 28.311552 MB
    print(f"MB per expert (codes+scales): {mb_per_exp:.3f} MB -> {mb_per_exp/4.8:.2f} us at 4.8 TB/s\n")
    print(f"{'load':10s} {'T':>3s} | {'MoE track: routes, dist, floor_us, max_tok/e':>44s} | {'Decode track: routes, dist, floor_us, max_tok/e':>47s}")
    for name, b, q in shapes:
        T = b * q
        m_r, m_d, m_max, m_ge9 = [], [], [], []
        d_r, d_d, d_max, d_ge9 = [], [], [], []
        for s in range(1, 33):
            r, d, c = moe_track_sample(T, 1000 + s)
            m_r.append(r); m_d.append(d); m_max.append(int(c.max())); m_ge9.append(int((c >= 9).sum()))
            r2, d2, c2 = decode_track_sample(b, q, 2000 + s)
            d_r.append(r2); d_d.append(d2); d_max.append(int(c2.max())); d_ge9.append(int((c2 >= 9).sum()))
        mr, md, mm, mg = sum(m_r)/len(m_r), sum(m_d)/len(m_d), sum(m_max)/len(m_max), sum(m_ge9)/len(m_ge9)
        dr, dd, dm, dg = sum(d_r)/len(d_r), sum(d_d)/len(d_d), sum(d_max)/len(d_max), sum(d_ge9)/len(d_ge9)
        print(f"{name:10s} {T:3d} | {mr:6.1f} r, {md:5.2f} exp ({md*mb_per_exp/4.8:5.1f}us), max={mm:5.1f}, >=9:{mg:4.1f}"
              f" | {dr:6.1f} r, {dd:5.2f} exp ({dd*mb_per_exp/4.8:5.1f}us), max={dm:5.1f}, >=9:{dg:4.1f}")


if __name__ == "__main__":
    summarize()
