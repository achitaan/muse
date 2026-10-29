
# Contiguous-KV FA3 selector for the full-attention layers, P6: rows gathered as 16 x int64
# (P4 gathered bytes) (lane omos/other-tracks-20261029).
# With page size 1, FA3 reads the paged FP8 cache through its non-TMA gather path. The full
# layers are the weakest Prefill shapes (1.43x; FA3 is 39% of turn 1 at 80k context). prepare()
# times, per scored full-layer shape, FA3 on a contiguous copy of the KV rows (gathered as bytes
# by the page table, gather included in the timing) against the incumbent paged call as cold-L2
# graph replays, and uses the gathered form only if it is >=3% faster with matching output.
_P4_SEL = set()
_P4_SEEN = set()
_P4_SHAPES = ((1, 753, 80000), (4, 753, 80000), (1, 16384, 0), (1, 16384, 32768), (1, 16384, 65536))
_p4_base = flash_attn_with_kvcache


def _p4_gathered(kw):
    (kc, vc, pt) = (kw['k_cache'], kw['v_cache'], kw['page_table'])
    (b, mp) = pt.shape
    idx = pt.reshape(-1).long().clamp_min(0)
    row = kc.shape[-2] * kc.shape[-1]
    wide = torch.int64 if row % 8 == 0 else torch.uint8
    kg = kc.view(torch.uint8).reshape(-1, row).view(wide).index_select(0, idx).view(torch.uint8).view(kc.dtype).view(b, mp, kc.shape[-2], kc.shape[-1])
    vg = vc.view(torch.uint8).reshape(-1, row).view(wide).index_select(0, idx).view(torch.uint8).view(vc.dtype).view(b, mp, vc.shape[-2], vc.shape[-1])
    return _p4_base(**dict(kw, k_cache=kg, v_cache=vg, page_table=None, scheduler_metadata=None))


def flash_attn_with_kvcache(*args, **kw):
    pt = kw.get('page_table')
    if pt is not None and not args and kw['k_cache'].dim() == 4 and kw['k_cache'].shape[1] == 1 and kw.get('window_size', (-1, -1))[0] < 0:
        key = (pt.shape[0], kw.get('max_seqlen_q'), pt.shape[1])
        if key not in _P4_SEEN:
            _P4_SEEN.add(key)
            print(f"P4_{'USE' if key in _P4_SEL else 'PAGED'} {key}", flush=True)
        if key in _P4_SEL:
            return _p4_gathered(kw)
    return _p4_base(*args, **kw)


def _p4_time(fn, flush, reps=7):
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


def _p4_select(dev):
    import time
    t0 = time.monotonic()
    rows = max(b * (c + l) for (b, l, c) in _P4_SHAPES)
    kc = torch.randn((rows, 1, 2, 64), device=dev).to(FP8)
    vc = torch.randn((rows, 1, 2, 64), device=dev).to(FP8)
    flush = torch.ones(64 << 20, dtype=torch.float32, device=dev)
    scale = 64 ** -0.5 * 5.4730064863875
    for (b, l, c) in _P4_SHAPES:
        mp = c + l
        pt = torch.randperm(b * mp, device=dev).to(torch.int32).view(b, mp)
        seq = torch.full((b,), mp, dtype=torch.int32, device=dev)
        cu = torch.arange(0, b * l + 1, l, dtype=torch.int32, device=dev)
        q = torch.randn((b * l, 16, 64), device=dev).to(FP8)
        kw = dict(q=q, k_cache=kc, v_cache=vc, page_table=pt, cache_seqlens=seq, cu_seqlens_q=cu,
                  max_seqlen_q=l, softmax_scale=scale, causal=True, window_size=(-1, -1), scheduler_metadata=None)
        first = lambda o: o[0] if isinstance(o, tuple) else o
        paged = lambda: first(_p4_base(**kw))
        gathered = lambda: first(_p4_gathered(kw))
        key = (b, l, mp)
        try:
            ref = paged().float()
            err = float((gathered().float() - ref).norm() / ref.norm().clamp_min(1e-30))
            base = _p4_time(paged, flush)
            t = _p4_time(gathered, flush)
            idx = pt.reshape(-1).long()
            tg = _p4_time(lambda: kc.view(torch.uint8).reshape(-1, 128).view(torch.int64).index_select(0, idx), flush)
            print(f'P4_GATHER {key} one-cache gather {tg:.1f}us', flush=True)
            base = min(base, _p4_time(paged, flush))
        except Exception as x:
            print(f'P4_SKIP {key} {type(x).__name__}: {x}', flush=True)
            continue
        ok = err < 1e-2 and t < 0.97 * base
        if ok:
            _P4_SEL.add(key)
        print(f"P4_CANDIDATE {key} gathered {t:.1f}us paged {base:.1f}us err {err:.2e} {'SELECT' if ok else 'KEEP'}", flush=True)
        del pt, seq, cu, q, ref
    del kc, vc, flush
    torch.cuda.synchronize()
    print(f'P4_SELECTOR selected={sorted(_P4_SEL)} s={time.monotonic() - t0:.1f}', flush=True)


_p4_original_prepare = prepare


def prepare(weights):
    state = _p4_original_prepare(weights)
    _P4_SEL.clear()
    try:
        _p4_select(weights['qkv_gate'].device)
    except Exception as x:
        _P4_SEL.clear()
        print(f'P4_SELECTOR off {type(x).__name__}: {x}', flush=True)
    torch.cuda.synchronize()
    return state
