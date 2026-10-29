
# olayer configuration selector for the Dense output projection at 128 and 256 tokens
# (lane omos/other-tracks-20261029). O at 256 tokens runs a fixed cuBLASLt algorithm at 9.27 us
# vs a 4.48 us floor (1.028x, the weakest Dense shape) and run_olayer stops at 224 tokens;
# O at 128 runs olayer (128,128,4,3,1,1) at 7.02 vs 3.99 us. prepare() times every compiled
# olayer configuration with a token tile >= 64 (multi-tile grids allowed; stores are m-guarded)
# against the incumbent as cold-L2 CUDA-graph replays and uses one only if it is >=3% faster with
# an FP64-reference error no worse than 1.5x the incumbent's.
def _d6_time(fn, flush, reps=11):
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


def _d6_select(state, weights):
    st = state.get('olayer') or {}
    if not st.get('enabled'):
        print('D6 olayer disabled', flush=True)
        return
    dev = weights['o'].device
    w = weights['o']
    (n, k) = w.shape
    flush = torch.ones(64 << 20, dtype=torch.float32, device=dev)
    gen = torch.Generator(device=dev).manual_seed(20261029)
    cands = [i for (i, c) in enumerate(st['configs']) if c[0] >= 64]
    for m in (128, 256):
        x = torch.randn((m, k), generator=gen, device=dev).to(torch.bfloat16)
        ref = x.double() @ w.double().T
        data = {'x': x, 'config': {'m': m, 'n': n, 'k': k, 'layer': 'o'}}
        base_fn = lambda: _d6_prev_kernel(data, state)
        err = lambda y: float((y.double() - ref).norm() / ref.norm())
        base_err = err(base_fn())
        base = _d6_time(base_fn, flush)
        best = None
        for i in cands:
            try:
                e = err(run_olayer_candidate(x, st, i))
                t = _d6_time(lambda: run_olayer_candidate(x, st, i), flush)
            except Exception as ex:
                print(f'D6_SKIP m{m} cfg{st["configs"][i]} {type(ex).__name__}', flush=True)
                continue
            print(f'D6_CANDIDATE m{m} cfg{st["configs"][i]} {t:.2f}us err {e:.2e} base {base:.2f}us err {base_err:.2e}', flush=True)
            if e <= 1.5 * base_err and (best is None or t < best[0]):
                best = (t, i)
        base = min(base, _d6_time(base_fn, flush))
        if best is not None and best[0] < 0.97 * base:
            state['d6_sel'][m] = best[1]
            print(f'D6_SELECT m{m} cfg{st["configs"][best[1]]} {best[0]:.2f} base {base:.2f}', flush=True)
        else:
            print(f'D6_KEEP m{m} base {base:.2f} best {None if best is None else round(best[0], 2)}', flush=True)
    del flush
    torch.cuda.synchronize()


_d6_prev_kernel = kernel_fn


def kernel_fn(data, state):
    sel = state.get('d6_sel')
    if sel:
        (x, cfg) = (data['x'], data['config'])
        i = sel.get(x.shape[0]) if cfg['layer'] == 'o' and cfg['n'] == 8192 else None
        if i is not None and x.dtype == torch.bfloat16 and x.is_contiguous() and x.data_ptr() % 16 == 0:
            return run_olayer_candidate(x, state['olayer'], i)
    return _d6_prev_kernel(data, state)


_d6_prev_prepare = prepare


def prepare(weights):
    state = _d6_prev_prepare(weights)
    state['d6_sel'] = {}
    try:
        _d6_select(state, weights)
    except Exception as ex:
        state['d6_sel'] = {}
        print(f'D6 off {type(ex).__name__}: {ex}', flush=True)
    torch.cuda.synchronize()
    return state
