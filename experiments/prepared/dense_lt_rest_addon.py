
# cuBLASLt neighborhood search for the remaining fixed M=128/256 descriptors: qkv_gate, pooler, o
# (D8; D7 covered the unpooler) (lane omos/other-tracks-20261029).
# verify 16 / verify 32 unpooler are the weakest memory-bound Dense shapes (1.030x / 1.045x;
# 19.28 vs 14.64 us and 23.64 vs 17.37 us floor). Their fixed descriptors in
# tt_lt__LT_H200_FIXED are restored, expanded (split-K 2-16, stage IDs, swizzle x custom x
# cluster) and joined by the first 16 exhaustive-heuristic algorithms of the same layout.
# prepare() times every variant against the incumbent as cold-L2 CUDA-graph replays and uses one
# only if it is >=3% faster with an FP64-reference error no worse than 1.5x the incumbent's.
def _d8_time(fn, flush, reps=11):
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


def _d8_select(state, weights):
    import time
    t0 = time.monotonic()
    dev = weights['qkv_gate'].device
    st = tt_lt__lt_state({name: weights[name] for name in tt_lt__LT_LAYERS})
    flush = torch.ones(64 << 20, dtype=torch.float32, device=dev)
    gen = torch.Generator(device=dev).manual_seed(20261029)
    for (m, layer) in ((256, 'qkv_gate'), (128, 'qkv_gate'), (256, 'pooler'), (128, 'pooler'), (256, 'o')):
        (layout, _, values) = tt_lt__LT_H200_FIXED[(m, layer)]
        w = weights[layer]
        (n, k) = w.shape
        x = torch.randn((m, k), generator=gen, device=dev).to(torch.bfloat16)
        ref = x.double() @ w.double().T
        data = {'x': x, 'config': {'m': m, 'n': n, 'k': k, 'layer': layer}}
        base_fn = lambda: _d8_prev_kernel(data, state)
        err = lambda y: float((y.double() - ref).norm() / ref.norm())
        base_err = err(base_fn())
        base = _d8_time(base_fn, flush)
        plan = tt_lt__lt_plan(st, layer, m, layout, 0)
        fixed = plan.restore(list(values))
        if fixed < 0:
            print(f'D8_SKIP m{m} {layer} restore failed', flush=True)
            continue
        plan.expand([fixed])
        n_exp = len(plan.metadata())
        plan.enumerate(True)
        total = min(len(plan.metadata()), n_exp + 16)
        out = torch.empty((m, n), dtype=torch.bfloat16, device=dev)
        (best, timed) = (None, 0)
        for idx in range(total):
            try:
                if plan.run(x, out, idx) != 0:
                    continue
                e = err(out)
                t = _d8_time(lambda: plan.run(x, out, idx), flush)
            except Exception as ex:
                continue
            timed += 1
            if idx == fixed:
                print(f'D8_FIXED m{m} {layer} {t:.2f}us base {base:.2f}us', flush=True)
            if e <= 1.5 * base_err and (best is None or t < best[0]):
                best = (t, idx, plan.metadata()[idx])
        base = min(base, _d8_time(base_fn, flush))
        print(f'D8_SEARCH m{m} {layer} timed {timed} of {total} (expanded {n_exp}) base {base:.2f}us err {base_err:.2e} best {None if best is None else round(best[0], 2)} {None if best is None else best[2]}', flush=True)
        if best is not None and best[0] < 0.97 * base:
            state['d8_sel'][(m, layer, n)] = (plan, best[1])
            print(f'D8_SELECT m{m} {layer} {best[0]:.2f} base {base:.2f} gain {base / best[0]:.4f}', flush=True)
        else:
            print(f'D8_KEEP m{m} {layer}', flush=True)
    state['d8_lt'] = st
    del flush
    torch.cuda.synchronize()
    print(f'D8_SELECTOR s={time.monotonic() - t0:.1f}', flush=True)


_d8_prev_kernel = kernel_fn


def kernel_fn(data, state):
    sel = state.get('d8_sel')
    if sel:
        (x, cfg) = (data['x'], data['config'])
        v = sel.get((x.shape[0], cfg['layer'], cfg['n']))
        if v is not None and x.dtype == torch.bfloat16 and x.is_contiguous() and x.data_ptr() % 256 == 0:
            out = torch.empty((x.shape[0], cfg['n']), dtype=torch.bfloat16, device=x.device)
            if v[0].run(x, out, v[1]) == 0:
                return out
    return _d8_prev_kernel(data, state)


_d8_prev_prepare = prepare


def prepare(weights):
    state = _d8_prev_prepare(weights)
    state['d8_sel'] = {}
    try:
        _d8_select(state, weights)
    except Exception as ex:
        state['d8_sel'] = {}
        print(f'D8_SELECTOR off {type(ex).__name__}: {ex}', flush=True)
    torch.cuda.synchronize()
    return state
