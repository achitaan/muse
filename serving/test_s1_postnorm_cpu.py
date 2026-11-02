"""CPU semantics test for serving change S1 (Marlin small-token post-expert norm -> fused Triton path).

Compares, for T in {1,4,8,16,31,32,33,64}, the stock small-token PyTorch fallback (copied verbatim from the
pinned fused_post_expert_norm_weighted_sum) against a line-by-line torch emulation of the Triton kernel
(_post_expert_norm_weighted_sum_kernel: skip slots with id < 0, FP32 weight-first RMSNorm, FP32 weighted sum,
BF16 store), and both against an FP64 reference. Remote (id < 0) slots are poisoned with NaN/Inf/1e30.
This checks formula/semantic equivalence only; it does NOT execute the Triton code or a GPU.
"""
import itertools, math, sys
import torch

H, TOPK, EPS = 4096, 8, 1e-8


def stock_fallback(x, s, w, ids, weight_first=True, apply_norm=True):
    tokens, topk, hidden = x.shape
    y = x
    if ids is not None:
        y = torch.where((ids.reshape(tokens, topk) >= 0).unsqueeze(-1), y, torch.zeros((), dtype=y.dtype))
    if apply_norm:
        f = y.float()
        if w is not None and weight_first:
            f = f * w.float(); f = f * torch.rsqrt(f.pow(2).mean(-1, keepdim=True) + EPS)
        elif w is not None:
            f = f * torch.rsqrt(f.pow(2).mean(-1, keepdim=True) + EPS); f = f * w.float()
        else:
            f = f * torch.rsqrt(f.pow(2).mean(-1, keepdim=True) + EPS)
        y = f
    else:
        y = y.float()
    y = y * s.float().unsqueeze(-1)
    return y.sum(dim=1).to(x.dtype)


def triton_emulation(x, s, w, ids, weight_first=True, apply_norm=True):
    tokens, topk, hidden = x.shape
    out = torch.empty((tokens, hidden), dtype=x.dtype)
    wf = w.float() if w is not None else None
    for t in range(tokens):
        acc = torch.zeros(hidden, dtype=torch.float32)
        for k in range(topk):
            if ids is not None and int(ids[t, k]) < 0:
                continue
            xv = x[t, k].float()
            if apply_norm:
                if wf is not None and weight_first:
                    y = xv * wf; var = (y * y).sum() / hidden; y = y * torch.rsqrt(var + EPS)
                elif wf is not None:
                    var = (xv * xv).sum() / hidden; y = xv * torch.rsqrt(var + EPS) * wf
                else:
                    var = (xv * xv).sum() / hidden; y = xv * torch.rsqrt(var + EPS)
            else:
                y = xv
            acc += y * s[t, k].float()
        out[t] = acc.to(x.dtype)
    return out


def reference64(x, s, w, ids, weight_first=True):
    xd = x.double(); m = (ids >= 0).unsqueeze(-1)
    xd = torch.where(m, xd, torch.zeros((), dtype=torch.float64))
    if weight_first:
        f = xd * w.double(); f = f * torch.rsqrt(f.pow(2).mean(-1, keepdim=True) + EPS)
    else:
        f = xd * torch.rsqrt(xd.pow(2).mean(-1, keepdim=True) + EPS) * w.double()
    return (f * s.double().unsqueeze(-1) * m).sum(1)


def make(T, pattern, g, scale=1.0):
    x = (torch.randn((T, TOPK, H), generator=g) * scale).to(torch.bfloat16)
    s = torch.rand((T, TOPK), generator=g); s = s / s.sum(-1, keepdim=True)
    ids = torch.randint(0, 32, (T, TOPK), generator=g, dtype=torch.int32)
    if pattern == 'all_remote': ids[:] = -1
    elif pattern == 'mixed': ids[torch.rand((T, TOPK), generator=g) < 0.75] = -1
    elif pattern.startswith('one_live_'):
        p = int(pattern.split('_')[-1]); ids[:] = -1; ids[:, p] = 3
    # poison every remote slot (Marlin leaves those rows uninitialized)
    poison = torch.tensor([float('nan'), float('inf'), -float('inf'), 1e30]).to(torch.bfloat16)
    rem = (ids < 0)
    for i, (t, k) in enumerate(rem.nonzero().tolist()):
        x[t, k] = poison[i % 4]
    return x, s, ids


def main():
    g = torch.Generator().manual_seed(1234)
    w = (1 + 0.1 * torch.randn(H, generator=g)).to(torch.bfloat16)
    n = 0; worst = {}
    for T, pattern, scale, wfirst in itertools.product((1, 4, 8, 16, 31, 32, 33, 64),
                                                       ['all_local', 'mixed', 'all_remote'] + [f'one_live_{p}' for p in range(8)],
                                                       (1.0, 1e-3, 300.0), (True, False)):
        if pattern.startswith('one_live_') and (scale != 1.0 or not wfirst):
            continue
        x, s, ids = make(T, pattern, g, scale)
        a = stock_fallback(x, s, w, ids, wfirst)
        b = triton_emulation(x, s, w, ids, wfirst)
        r = reference64(x, s, w, ids, wfirst)
        for name, v in (('stock', a), ('fused', b)):
            assert torch.isfinite(v.float()).all(), (T, pattern, scale, name, 'non-finite')
            if pattern == 'all_remote':
                assert (v.float() == 0).all(), (T, pattern, name, 'all-remote must be exactly zero')
            err = ((v.double() - r).norm() / r.norm().clamp_min(1e-300)).item() if r.norm() > 0 else 0.0
            assert err < 8e-3, (T, pattern, scale, wfirst, name, err)
            worst[name] = max(worst.get(name, 0.0), err)
        d = ((a.float() - b.float()).abs().max()).item()
        worst['stock_vs_fused_maxabs'] = max(worst.get('stock_vs_fused_maxabs', 0.0), d)
        n += 1
    print(f'PASS {n} cases; worst rel-L2 vs FP64: stock {worst["stock"]:.2e}, fused-emulation {worst["fused"]:.2e}; '
          f'max |stock - fused| {worst["stock_vs_fused_maxabs"]:.3g} (BF16 rounding/reduction order)')


if __name__ == '__main__':
    sys.exit(main())
