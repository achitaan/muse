
# Row-parallel GEMV for the smallest Dense projections (lane omos/other-tracks-20261029).
# The router (FP32 256x8192) and the sliced unpooler (BF16 1024x4096) at 1 and 8 tokens are
# latency-bound. Each program loads its whole contiguous weight rows at once, accumulates
# in FP32 and rounds once to the output dtype. prepare() times row/warp variants against the
# incumbent as cold-L2 CUDA-graph replays and uses one only if it is >=3% faster and its
# FP64-reference error is no worse than the incumbent's; otherwise the incumbent runs.
import triton as _gv_triton
import triton.language as _gv_tl


@_gv_triton.jit
def _gvk(X, W, O, N, K: _gv_tl.constexpr, M: _gv_tl.constexpr, ROWS: _gv_tl.constexpr):
    pid = _gv_tl.program_id(0)
    offs = _gv_tl.arange(0, K)
    for r in _gv_tl.static_range(ROWS):
        row = pid * ROWS + r
        w = _gv_tl.load(W + row * K + offs).to(_gv_tl.float32)
        for m in _gv_tl.static_range(M):
            x = _gv_tl.load(X + m * K + offs).to(_gv_tl.float32)
            _gv_tl.store(O + m * N + row, _gv_tl.sum(w * x, axis=0))


def _gv_run(x, layer, n, w, rows, warps):
    out = torch.empty((x.shape[0], n), dtype=torch.float32 if layer == 'router' else torch.bfloat16, device=x.device)
    _gvk[(n // rows,)](x, w, out, n, K=x.shape[1], M=x.shape[0], ROWS=rows, num_warps=warps)
    return out


def _gv_time(fn, flush, reps=11):
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


def _gv_select(state, weights):
    import time
    t0 = time.monotonic()
    dev = weights['qkv_gate'].device
    state['gv_w'] = {'router': weights['router'].contiguous(), 'unpooler': weights['unpooler'][:1024].contiguous()}
    flush = torch.ones(64 << 20, dtype=torch.float32, device=dev)
    gen = torch.Generator(device=dev).manual_seed(20261029)
    variants = {'router': ((1, 8), (1, 4), (2, 8)), 'unpooler': ((4, 8), (2, 8), (4, 4), (8, 8))}
    for (layer, m) in (('router', 1), ('router', 8), ('unpooler', 1), ('unpooler', 8)):
        w = state['gv_w'][layer]
        (n, k) = w.shape
        x = torch.randn((m, k), generator=gen, device=dev).to(torch.bfloat16)
        ref = x.double() @ w.double().T
        data = {'x': x, 'config': {'m': m, 'n': n, 'k': k, 'layer': layer}}
        base_fn = lambda: _gv_prev_kernel(data, state)
        err = lambda y: float((y.double() - ref).norm() / ref.norm())
        base_err = err(base_fn())
        base = _gv_time(base_fn, flush)
        best = None
        for (rows, warps) in variants[layer]:
            fn = lambda: _gv_run(x, layer, n, w, rows, warps)
            try:
                e = err(fn())
                t = _gv_time(fn, flush)
            except Exception as ex:
                print(f'GV_SKIP {layer} m{m} {rows},{warps} {type(ex).__name__}', flush=True)
                continue
            ok = e <= max(1.5 * base_err, 1e-6)
            print(f'GV_CANDIDATE {layer} m{m} rows{rows} warps{warps} {t:.2f}us err {e:.2e} base {base:.2f}us err {base_err:.2e}', flush=True)
            if ok and (best is None or t < best[0]):
                best = (t, rows, warps)
        base = min(base, _gv_time(base_fn, flush))
        if best is not None and best[0] < 0.97 * base:
            state['gv_sel'][(m, layer, n)] = (best[1], best[2])
            print(f'GV_SELECT {layer} m{m} rows{best[1]} warps{best[2]} {best[0]:.2f} base {base:.2f}', flush=True)
        else:
            print(f'GV_KEEP {layer} m{m} base {base:.2f} best {None if best is None else round(best[0], 2)}', flush=True)
    del flush
    torch.cuda.synchronize()
    print(f'GV_SELECTOR selected={sorted(state["gv_sel"])} s={time.monotonic() - t0:.1f}', flush=True)


_gv_prev_kernel = kernel_fn


def kernel_fn(data, state):
    sel = state.get('gv_sel')
    if sel:
        (x, cfg) = (data['x'], data['config'])
        v = sel.get((x.shape[0], cfg['layer'], cfg['n']))
        if v is not None and x.dtype == torch.bfloat16 and x.is_contiguous() and x.shape[1] == state['gv_w'][cfg['layer']].shape[1]:
            return _gv_run(x, cfg['layer'], cfg['n'], state['gv_w'][cfg['layer']], v[0], v[1])
    return _gv_prev_kernel(data, state)


_gv_prev_prepare = prepare


def prepare(weights):
    state = _gv_prev_prepare(weights)
    state['gv_sel'] = {}
    try:
        _gv_select(state, weights)
    except Exception as ex:
        state['gv_sel'] = {}
        print(f'GV_SELECTOR off {type(ex).__name__}: {ex}', flush=True)
    torch.cuda.synchronize()
    return state
