
# olayer configuration selector for the Dense output projection at 1/8/16/32 tokens
# (lane omos/other-tracks-20261029). The O projection runs 1.4-1.5x its floor at these sizes.
# prepare() times every already-compiled olayer configuration whose token tile is 16, 32 or
# 64 against the incumbent dispatch as cold-L2 CUDA-graph replays and uses one only if it is
# >=3% faster with an FP64-reference error no worse than 1.5x the incumbent's.
def _d5_time(fn, flush, reps=11):
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


def _d5_select(state, weights):
    st = state.get('olayer') or {}
    if not st.get('enabled'):
        print('D5 olayer disabled', flush=True)
        return
    dev = weights['o'].device
    w = weights['o']
    (n, k) = w.shape
    flush = torch.ones(64 << 20, dtype=torch.float32, device=dev)
    gen = torch.Generator(device=dev).manual_seed(20261029)
    cands = [i for (i, c) in enumerate(st['configs']) if c[0] in (16, 32, 64)]
    for m in (1, 8, 16, 32):
        x = torch.randn((m, k), generator=gen, device=dev).to(torch.bfloat16)
        ref = x.double() @ w.double().T
        data = {'x': x, 'config': {'m': m, 'n': n, 'k': k, 'layer': 'o'}}
        base_fn = lambda: _d5_prev_kernel(data, state)
        err = lambda y: float((y.double() - ref).norm() / ref.norm())
        base_err = err(base_fn())
        base = _d5_time(base_fn, flush)
        best = None
        for i in cands:
            if st['configs'][i][0] < m:
                continue
            try:
                e = err(run_olayer_candidate(x, st, i))
                t = _d5_time(lambda: run_olayer_candidate(x, st, i), flush)
            except Exception as ex:
                print(f'D5_SKIP m{m} cfg{st["configs"][i]} {type(ex).__name__}', flush=True)
                continue
            print(f'D5_CANDIDATE m{m} cfg{st["configs"][i]} {t:.2f}us err {e:.2e} base {base:.2f}us err {base_err:.2e}', flush=True)
            if e <= 1.5 * base_err and (best is None or t < best[0]):
                best = (t, i)
        base = min(base, _d5_time(base_fn, flush))
        if best is not None and best[0] < 0.97 * base:
            state['d5_sel'][m] = best[1]
            print(f'D5_SELECT m{m} cfg{st["configs"][best[1]]} {best[0]:.2f} base {base:.2f}', flush=True)
        else:
            print(f'D5_KEEP m{m} base {base:.2f} best {None if best is None else round(best[0], 2)}', flush=True)
    del flush
    torch.cuda.synchronize()


_d5_prev_kernel = kernel_fn


def kernel_fn(data, state):
    sel = state.get('d5_sel')
    if sel:
        (x, cfg) = (data['x'], data['config'])
        i = sel.get(x.shape[0]) if cfg['layer'] == 'o' and cfg['n'] == 8192 else None
        if i is not None and x.dtype == torch.bfloat16 and x.is_contiguous() and x.data_ptr() % 16 == 0:
            return run_olayer_candidate(x, state['olayer'], i)
    return _d5_prev_kernel(data, state)


_d5_prev_prepare = prepare


def prepare(weights):
    state = _d5_prev_prepare(weights)
    state['d5_sel'] = {}
    try:
        _d5_select(state, weights)
    except Exception as ex:
        state['d5_sel'] = {}
        print(f'D5 off {type(ex).__name__}: {ex}', flush=True)
    torch.cuda.synchronize()
    return state
