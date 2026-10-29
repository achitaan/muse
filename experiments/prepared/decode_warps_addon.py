
# Warp-count selector for the two single-row Triton boundaries (lane omos/decode-fresh).
_D4W = {}


def _sfn_boundary(stream, branch, state):
    cached = state['sliced_ffn_boundary']
    if (not branch.shape[0] or cached is None or not branch.is_contiguous()
            or not stream.is_contiguous()):
        padded = F.pad(branch, (0, HIDDEN - UNPOOL_SLICE))
        output, updated, _ = state['ffn_boundary'](stream, padded, branch_dtype=torch.bfloat16)
        return output, updated
    output = torch.empty(stream.shape, dtype=torch.bfloat16, device=stream.device)
    updated = torch.empty_like(stream)
    _sfn_kernel[(branch.shape[0],)](
        branch, stream, cached['post_weight'], cached['pre_weight'], cached['alpha'], cached['beta'],
        output, updated, branch.stride(0), stream.stride(0), HIDDEN, UNPOOL_SLICE,
        cached['post_eps'], cached['pre_eps'], num_warps=_D4W.get(('s', branch.shape[0]), 8), enable_fp_fusion=False)
    return output, updated


def _pn_expert_inputs(x, ids, local_start, state):
    rows, hidden = x.shape
    gain = state['pre_expert_norm']._get_effective_weight()
    assert x.dtype == gain.dtype == torch.bfloat16 and hidden == 4096
    out = torch.empty((rows, hidden), dtype=x.dtype, device=x.device)
    local_ids = torch.empty(ids.shape, dtype=torch.int32, device=ids.device)
    if rows:
        _pn_inputs_kernel[(rows,)](
            x, gain, ids, local_start, out, local_ids, x.stride(0), ids.stride(0),
            hidden, RMS_EPS, _pn_triton.next_power_of_2(hidden), num_warps=_D4W.get(('p', rows), 4),
            enable_fp_fusion=False)
    return out, local_ids


def _d4_time(fn, flush):
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    t = []
    for _ in range(11):
        flush.sum()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record(); g.replay(); e1.record(); e1.synchronize()
        t.append(e0.elapsed_time(e1) * 1000.0)
    return sorted(t)[5]


def _d4_select(state):
    dev = state['router'].device
    flush = torch.ones(32 << 20, dtype=torch.float32, device=dev)
    g = torch.Generator(device=dev).manual_seed(29)
    start = torch.zeros(1, dtype=torch.int32, device=dev)
    jobs = [('s', T, (4, 8, 16), 8) for T in (1, 8)] + [('p', T, (2, 4, 8, 16), 4) for T in (1, 8, 16, 32, 128, 256)]
    for (kind, T, ws, inc) in jobs:
        if kind == 's':
            a = (torch.randn((T, HIDDEN), generator=g, device=dev), torch.randn((T, UNPOOL_SLICE), generator=g, device=dev).to(torch.bfloat16))
            run = lambda: _sfn_boundary(a[0], a[1], state)
        else:
            a = (torch.randn((T, 4096), generator=g, device=dev).to(torch.bfloat16), torch.randint(0, 256, (T, 8), generator=g, device=dev, dtype=torch.int32))
            run = lambda: _pn_expert_inputs(a[0], a[1], start, state)
        res = {}
        for w in ws:
            _D4W[(kind, T)] = w
            res[w] = ([o.float().clone() for o in run()], _d4_time(run, flush))
        _D4W.pop((kind, T), None)
        ref = res[inc][0]
        ok = {w: all(float((o - r).norm() / r.norm().clamp_min(1e-30)) < 1e-3 for (o, r) in zip(res[w][0], ref)) for w in ws}
        best = min((w for w in ws if ok[w]), key=lambda w: res[w][1])
        print(f'D4 {kind} T={T} ' + ' '.join(f'w{w}={res[w][1]:.2f}' for w in ws) + f' incumbent w{inc}', flush=True)
        if best != inc and res[best][1] < 0.97 * res[inc][1]:
            _D4W[(kind, T)] = best
    print(f'D4_SELECT {sorted(_D4W.items())}', flush=True)


_d4_prev_prepare = prepare


def prepare(weights):
    state = _d4_prev_prepare(weights)
    _D4W.clear()
    try:
        _d4_select(state)
    except Exception as ex:
        _D4W.clear()
        print(f'D4 off {type(ex).__name__}: {ex}', flush=True)
    torch.cuda.synchronize()
    return state
