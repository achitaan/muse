
# cuBLASLt algorithm selector for the one M=128/256 Dense shape without a fixed descriptor,
# (128, 'o') (lane omos/other-tracks-20261029). prepare() enumerates cuBLASLt heuristic
# algorithms in each weight layout through the existing tt_lt extension, times each against
# the incumbent as cold-L2 CUDA-graph replays, and uses one only if it is >=3% faster with an
# FP64-reference error no worse than 1.5x the incumbent's.
def _d3_time(fn, flush, reps=11):
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    t = []
    for _ in range(reps):
        flush.sum()
        (e0, e1) = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        e0.record()
        g.replay()
        e1.record()
        e1.synchronize()
        t.append(e0.elapsed_time(e1) * 1000.0)
    del g
    return sorted(t)[reps // 2]


def _d3_select(state, weights):
    import time
    t0 = time.monotonic()
    dev = weights['qkv_gate'].device
    (m, layer) = (128, 'o')
    w = weights[layer].contiguous()
    (n, k) = w.shape
    st = tt_lt__lt_state({name: weights[name] for name in tt_lt__LT_LAYERS})
    flush = torch.ones(64 << 20, dtype=torch.float32, device=dev)
    gen = torch.Generator(device=dev).manual_seed(20261029)
    x = torch.randn((m, k), generator=gen, device=dev).to(torch.bfloat16)
    ref = x.double() @ w.double().T
    data = {'x': x, 'config': {'m': m, 'n': n, 'k': k, 'layer': layer}}
    base_fn = lambda: _d3_prev_kernel(data, state)
    err = lambda y: float((y.double() - ref).norm() / ref.norm())
    base_err = err(base_fn())
    base = _d3_time(base_fn, flush)
    out = torch.empty((m, n), dtype=torch.bfloat16, device=dev)
    best = None
    for layout in (0, 1, 2, 3):
        try:
            plan = tt_lt__lt_plan(st, layer, m, layout, 0)
            plan.enumerate(False)
            count = len(plan.metadata())
        except Exception as ex:
            print(f'LT_SKIP layout{layout} {type(ex).__name__}', flush=True)
            continue
        for idx in range(min(count, 8)):
            try:
                if plan.run(x, out, idx) != 0:
                    continue
                e = err(out)
                t = _d3_time(lambda: plan.run(x, out, idx), flush)
            except Exception as ex:
                print(f'LT_SKIP layout{layout} algo{idx} {type(ex).__name__}', flush=True)
                continue
            print(f'LT_CANDIDATE o m128 layout{layout} algo{idx} {t:.2f}us err {e:.2e} base {base:.2f}us err {base_err:.2e}', flush=True)
            if e <= 1.5 * base_err and (best is None or t < best[0]):
                best = (t, plan, idx, layout)
    base = min(base, _d3_time(base_fn, flush))
    if best is not None and best[0] < 0.97 * base:
        state['d3_sel'] = {(m, layer, n): (best[1], best[2])}
        state['d3_lt'] = st
        print(f'LT_SELECT o m128 layout{best[3]} algo{best[2]} {best[0]:.2f} base {base:.2f}', flush=True)
    else:
        print(f'LT_KEEP o m128 base {base:.2f} best {None if best is None else round(best[0], 2)}', flush=True)
    del flush
    torch.cuda.synchronize()
    print(f'LT_SELECTOR s={time.monotonic() - t0:.1f}', flush=True)


_d3_prev_kernel = kernel_fn


def kernel_fn(data, state):
    sel = state.get('d3_sel')
    if sel:
        (x, cfg) = (data['x'], data['config'])
        v = sel.get((x.shape[0], cfg['layer'], cfg['n']))
        if v is not None and x.dtype == torch.bfloat16 and x.is_contiguous() and x.data_ptr() % 256 == 0:
            out = torch.empty((x.shape[0], cfg['n']), dtype=torch.bfloat16, device=x.device)
            if v[0].run(x, out, v[1]) == 0:
                return out
    return _d3_prev_kernel(data, state)


_d3_prev_prepare = prepare


def prepare(weights):
    state = _d3_prev_prepare(weights)
    state['d3_sel'] = {}
    try:
        _d3_select(state, weights)
    except Exception as ex:
        state['d3_sel'] = {}
        print(f'LT_SELECTOR off {type(ex).__name__}: {ex}', flush=True)
    torch.cuda.synchronize()
    return state
