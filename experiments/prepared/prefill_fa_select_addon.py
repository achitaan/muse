
# FA3 split / GQA-packing selector (lane omos/other-tracks-20261029).
# prepare() times FA3's own num_splits/pack_gqa options as CUDA-graph replays with a
# cold L2 on synthetic inputs shaped like each scored attention call (page size 1,
# scattered pages, fixed lengths), and keeps one per shape only when it beats FA3's
# heuristic by >=3% and matches its output. Numerics stay FA3's own; any other shape,
# and any shape without a winner, calls FA3 exactly as before.
_FA_SEL = {}
_FA_SHAPES = ((1, 753, 2048, 80000), (1, 753, 0, 80000), (4, 753, 2048, 80000), (4, 753, 0, 80000),
              (1, 16384, 0, 0), (1, 16384, 2048, 32768), (1, 16384, 0, 32768), (1, 16384, 0, 65536))
_fa_base = flash_attn_with_kvcache


def flash_attn_with_kvcache(*args, **kw):
    pt = kw.get('page_table')
    if pt is not None and _FA_SEL:
        sel = _FA_SEL.get((pt.shape[0], kw.get('max_seqlen_q'), kw.get('window_size', (-1, -1))[0] >= 0, pt.shape[1]))
        if sel:
            kw = dict(kw, **sel)
    return _fa_base(*args, **kw)


def _fa_time(fn, flush, reps=7):
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


def _fa_select(dev):
    import time
    t0 = time.monotonic()
    rows = max(b * (c + l) for (b, l, _, c) in _FA_SHAPES)
    kc = torch.randn((rows, 1, 2, 64), device=dev).to(FP8)
    vc = torch.randn((rows, 1, 2, 64), device=dev).to(FP8)
    flush = torch.ones(64 << 20, dtype=torch.float32, device=dev)
    scale = 64 ** -0.5 * 5.4730064863875
    for (b, l, w, c) in _FA_SHAPES:
        mp = c + l
        pt = torch.randperm(b * mp, device=dev).to(torch.int32).view(b, mp)
        seq = torch.full((b,), mp, dtype=torch.int32, device=dev)
        cu = torch.arange(0, b * l + 1, l, dtype=torch.int32, device=dev)
        q = torch.randn((b * l, 16, 64), device=dev).to(FP8)
        win = (w - 1, 0) if w else (-1, -1)

        def call(extra):
            out = _fa_base(q=q, k_cache=kc, v_cache=vc, page_table=pt, cache_seqlens=seq, cu_seqlens_q=cu,
                           max_seqlen_q=l, softmax_scale=scale, causal=True, window_size=win,
                           scheduler_metadata=None, **extra)
            return out[0] if isinstance(out, tuple) else out
        key = (b, l, w > 0, mp)
        ref = call({}).float()
        base = _fa_time(lambda: call({}), flush)
        best = None
        for ns in (0, 1, 2, 3, 4, 8):
            for pg in (None, True, False):
                if ns == 0 and pg is None:
                    continue
                extra = {'num_splits': ns} if pg is None else {'num_splits': ns, 'pack_gqa': pg}
                try:
                    err = float((call(extra).float() - ref).norm() / ref.norm().clamp_min(1e-30))
                    t = _fa_time(lambda: call(extra), flush)
                except Exception as x:
                    print(f'FA_SKIP {key} {extra} {type(x).__name__}', flush=True)
                    continue
                print(f'FA_CANDIDATE {key} {extra} {t:.2f} err {err:.2e} base {base:.2f}', flush=True)
                if err < 1e-2 and (best is None or t < best[0]):
                    best = (t, extra)
        base = min(base, _fa_time(lambda: call({}), flush))
        if best is not None and best[0] < 0.97 * base:
            _FA_SEL[key] = best[1]
            print(f'FA_SELECT {key} {best[1]} {best[0]:.2f} base {base:.2f}', flush=True)
        else:
            print(f'FA_KEEP {key} base {base:.2f} best {None if best is None else round(best[0], 2)}', flush=True)
        del pt, seq, cu, q, ref
    del kc, vc, flush
    torch.cuda.synchronize()
    print(f'FA_SELECTOR selected={len(_FA_SEL)} s={time.monotonic() - t0:.1f}', flush=True)


_fa_original_prepare = prepare


def prepare(weights):
    state = _fa_original_prepare(weights)
    _FA_SEL.clear()
    try:
        _fa_select(weights['qkv_gate'].device)
    except Exception as x:
        _FA_SEL.clear()
        print(f'FA_SELECTOR off {type(x).__name__}: {x}', flush=True)
    torch.cuda.synchronize()
    return state
