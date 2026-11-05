# Prepare-only mathematical reference. Adapted from the public task's
# reference.py / nvfp4.py / checks.py (Open Kernel Challenge, Apache-2.0).
# Only B1, L1/8, page1 is used here. No task/evaluator imports at runtime.


def _dc_ref_rms(x, eps=1e-5):
    x = x.float()
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def _dc_ref_rope(x, pos, theta):
    inv = 1. / (theta ** (torch.arange(0, 64, 2, device=x.device).float() / 64))
    ang = pos.float()[:, None] * inv
    cos, sin = ang.cos()[:, None, :], ang.sin()[:, None, :]
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack([even * cos - odd * sin, even * sin + odd * cos], -1).flatten(-2)


def _dc_ref_coeff(gate):
    w = gate.float() / .3
    beta = torch.sigmoid(w)
    alpha = torch.sqrt(torch.clamp(torch.sigmoid(-w) * (1. + beta), min=1e-3))
    return alpha, beta


def _dc_ref_attention(data, weights, query_dtype):
    cfg = data['config']
    length = cfg['q_len']
    qkv = data['attn_input'].float() @ weights['qkv_gate'].float().T
    q, k, v, gate = qkv.split([1024, 128, 128, 1024], dim=-1)
    pos = data['seq_lens'].long() + torch.arange(length, device=qkv.device)
    q, k = _dc_ref_rms(q.view(-1, 16, 64)), _dc_ref_rms(k.view(-1, 2, 64))
    if cfg['rope_theta']:
        q, k = _dc_ref_rope(q, pos, cfg['rope_theta']), _dc_ref_rope(k, pos, cfg['rope_theta'])
    if query_dtype == torch.bfloat16:
        q = (q * cfg['qk_scale_factor']).to(torch.bfloat16).float()
    else:
        q = q.to(query_dtype).float() * cfg['qk_scale_factor']
    kn = (k / data['k_scale']).to(torch.float8_e4m3fn)
    vn = (v.reshape(-1, 2, 64) / data['v_scale']).to(torch.float8_e4m3fn)
    seq = int(data['seq_lens'][0])
    first = max(0, seq - cfg['window'] + 1) if cfg['window'] else 0
    past = torch.arange(first, seq, device=qkv.device)
    rows = data['page_table'][0].long()[past]
    # CPU/GPU torch supports advanced FP8 indexing just as the task reference.
    keys = torch.cat([data['k_cache'].view(-1, 2, 64)[rows].float() * data['k_scale'], kn.float() * data['k_scale']])
    vals = torch.cat([data['v_cache'].view(-1, 2, 64)[rows].float() * data['v_scale'], vn.float() * data['v_scale']])
    keypos = torch.cat([past, seq + torch.arange(length, device=qkv.device)])
    visible = keypos[None, :] <= pos[:, None]
    if cfg['window']:
        visible &= keypos[None, :] > pos[:, None] - cfg['window']
    att = torch.einsum('lkgd,tkd->lkgt', q.reshape(length, 2, 8, 64), keys) * cfg['sm_scale']
    att = att.masked_fill(~visible[:, None, None, :], float('-inf')).softmax(-1)
    out = torch.einsum('lkgt,tkd->lkgd', att, vals).reshape(length, 16, 64)
    out = _dc_ref_rms(out).reshape(length, 1024) * torch.sigmoid(gate)
    return out, kn.view(1, length, 2, 64), vn.view(1, length, 2, 64)


def _dc_ref_front(data, weights, query_dtype=torch.bfloat16):
    out, kn, vn = _dc_ref_attention(data, weights, query_dtype)
    a = out @ weights['o'].float().T
    alpha, beta = _dc_ref_coeff(weights['attn_residual_gate'])
    stream = alpha * data['stream'].float() + beta * _dc_ref_rms(a, 1e-8)
    h = _dc_ref_rms(stream) * (1. + weights['pre_ffn_norm'].float())
    logits = h.double() @ weights['router'].double().T
    scores = torch.sigmoid(logits * .1767766952966369).float()
    return dict(stream1=stream, h=h, scores=scores, biased=scores + weights['router_bias'].float(),
                k_new=kn, v_new=vn, local_start=int(data['local_start']))


def _dc_ref_dequant(packed, sf, gs):
    p = packed.view(torch.uint8)
    codes = torch.stack((p & 15, (p >> 4) & 15), dim=-1).flatten(-2)
    grid = torch.tensor((0., .5, 1., 1.5, 2., 3., 4., 6.), device=p.device, dtype=torch.float32)
    mag = grid[(codes & 7).long()]
    vals = torch.where((codes & 8) != 0, -mag, mag)
    blocks = vals.view(*vals.shape[:-1], vals.shape[-1] // 16, 16)
    return (blocks * (sf.float() * gs).unsqueeze(-1)).view_as(vals)


def _dc_ref_back(front, weights, ids, deadline):
    picked = front['scores'].gather(1, ids.long())
    wt = picked / (picked.sum(-1, keepdim=True) + 1e-15)
    p = front['h'] @ weights['pooler'].float().T
    p = _dc_ref_rms(p) * (1. + weights['pre_expert_norm'].float())
    local = (ids.long() - front['local_start']) % 256
    out = torch.zeros_like(p)
    for e in torch.unique(local).tolist():
        if not 0 <= e < 32:
            continue
        _dc_budget(deadline, 1)
        tok, slot = (local == e).nonzero(as_tuple=True)
        w13 = _dc_ref_dequant(weights['w13'][e], weights['w13_sf'][e], weights['w13_gs'][e])
        h = p[tok] @ w13.T
        del w13
        inter = F.silu(h[:, :4096]) * h[:, 4096:]
        w2 = _dc_ref_dequant(weights['w2'][e], weights['w2_sf'][e], weights['w2_gs'][e])
        y = (inter @ w2.T) * (1. + weights['post_expert_norm'].float())
        del w2
        y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + 1e-8)
        out.index_add_(0, tok, y * wt[tok, slot, None])
    f = torch.zeros_like(front['stream1'])
    f[:, :1024] = out @ weights['unpooler'][:1024].float().T
    f = f * (1. + weights['post_ffn_norm'].float())
    alpha, beta = _dc_ref_coeff(weights['ffn_residual_gate'])
    stream = alpha * front['stream1'] + beta * _dc_ref_rms(f, 1e-8)
    hidden = _dc_ref_rms(stream) * (1. + weights['next_pre_attn_norm'].float())
    return hidden.to(torch.bfloat16), stream


def _dc_reference(data, weights, front, ids, deadline=float('inf')):
    h, s = _dc_ref_back(front, weights, ids, deadline)
    fpfront = _dc_ref_front(data, weights, torch.float8_e4m3fn)
    fh, fs = _dc_ref_back(fpfront, weights, ids, deadline)
    a1, _ = _dc_ref_coeff(weights['attn_residual_gate'])
    a2, _ = _dc_ref_coeff(weights['ffn_residual_gate'])
    carry = a1 * a2 * data['stream'].float()
    update = s - carry
    # Same FP8-query calibration, clamping and row factors as the public check.
    bound = lambda err: min(.15, max(.03, 2.5 * err))
    hu = bound(_dc_metrics(fh, h)[0])
    su = bound(_dc_metrics(fs - carry, update)[0])
    vals = dict(hidden=h, stream=s, update=update, k_new=front['k_new'], v_new=front['v_new'])
    return vals, dict(hidden=hu, stream=su, update=su, k_new=.08, v_new=.08), carry


def _dc_ref_routing(ids, biased, local_start):
    if bool((ids < 0).any() | (ids >= 256).any()):
        return False
    ranked = ids.long().sort(-1).values
    if bool((ranked[:, 1:] == ranked[:, :-1]).any()):
        return False
    best = biased.topk(8, dim=-1)
    if float((best.values[:, -1:] - biased.gather(1, ids.long())).max()) > .05:
        return False
    ninth = biased.topk(9, dim=-1).values[:, -1:]
    kept = (best.indices[:, :, None] == ids.long()[:, None, :]).any(-1)
    local = (best.indices - local_start) % 256 < 32
    if bool((local & ~kept & (best.values - ninth > .03)).any()):
        return False
    def load(ix):
        local = ix[(ix - local_start) % 256 < 32]
        return local.numel(), torch.unique(local).numel()
    nr, nd = load(ids)
    rr, rd = load(best.indices)
    return nr >= rr * .92 - 4 and nd >= rd * .85 - 2
