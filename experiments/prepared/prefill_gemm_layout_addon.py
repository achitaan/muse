
# Dense-GEMM layout selector (lane omos/other-tracks-20261029). The QKV/O/pooler/unpooler
# GEMMs are ~4.2 ms of the 9.6 ms first-turn shape (cuBLAS, ~77% of BF16 peak). prepare()
# times, per scored token count and layer, the incumbent x @ W^T call against torch.mm with a
# pre-transposed copy of the same BF16 weight (a different cuBLAS layout and kernel choice),
# as cold-L2 graph replays, and uses the transposed form only when it is >=3% faster with
# output matching the incumbent. Weights keep their BF16 values; nothing else changes.
_P3_SEL = {}
_p3_gemm_base = _gemm


def _gemm(x, w):
    v = _P3_SEL.get((x.shape[0], w.data_ptr(), tuple(w.shape)))
    if v is not None and x.is_contiguous():
        return torch.mm(x, v)
    return _p3_gemm_base(x, w)


def _p3_time(fn, flush, reps=7):
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


def _p3_select(state):
    import time
    t0 = time.monotonic()
    dense = state['dense']
    dev = dense['qkv_gate'].device
    flush = torch.ones(64 << 20, dtype=torch.float32, device=dev)
    gen = torch.Generator(device=dev).manual_seed(20261029)
    for name in ('qkv_gate', 'o', 'pooler', 'unpooler'):
        w = dense[name]
        wt = w.t().contiguous()
        used = False
        for m in (753, 3012, 16384):
            x = torch.randn((m, w.shape[1]), generator=gen, device=dev).to(torch.bfloat16)
            ref = _p3_gemm_base(x, w).float()
            alt = torch.mm(x, wt).float()
            err = float((alt - ref).norm() / ref.norm().clamp_min(1e-30))
            base = _p3_time(lambda: _p3_gemm_base(x, w), flush)
            t = _p3_time(lambda: torch.mm(x, wt), flush)
            base = min(base, _p3_time(lambda: _p3_gemm_base(x, w), flush))
            print(f'P3_CANDIDATE {name} m{m} transposed {t:.1f}us base {base:.1f}us err {err:.2e}', flush=True)
            if err < 1e-2 and t < 0.97 * base:
                _P3_SEL[(m, w.data_ptr(), tuple(w.shape))] = wt
                used = True
                print(f'P3_SELECT {name} m{m} gain {base / t:.4f}', flush=True)
            del x, ref, alt
        if not used:
            del wt
    del flush
    torch.cuda.synchronize()
    print(f'P3_SELECTOR selected={len(_P3_SEL)} s={time.monotonic() - t0:.1f}', flush=True)


_p3_original_prepare = prepare


def prepare(weights):
    state = _p3_original_prepare(weights)
    _P3_SEL.clear()
    try:
        _p3_select(state)
    except Exception as ex:
        _P3_SEL.clear()
        print(f'P3_SELECTOR off {type(ex).__name__}: {ex}', flush=True)
    torch.cuda.synchronize()
    return state
