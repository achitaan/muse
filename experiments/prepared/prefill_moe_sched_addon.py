
# Prefill MoE schedule selector by token count (lane omos/other-tracks-20261029).
# MoE is 37-44% of every Prefill shape. Its schedule follows two token thresholds
# (large mode at _pg_MOE_LARGE_T, fine ordering at _pg_FINE_MIN_T). prepare() times all four
# large x fine combinations at each scored token count (753, 3012, 16384) as cold-L2 graph
# replays on routing drawn like the task inputs, and overrides the thresholds for that token
# count only when a combination is >=3% faster with matching output. Kernels unchanged.
_PG_SEL = {}
_pg_moe_forward_base = _pg__moe_forward


def _pg__moe_forward(st, p, lid, pos, cnt, T, *a, **k):
    global _pg_MOE_LARGE_T, _pg_FINE_MIN_T
    sel = _PG_SEL.get(T)
    if sel is None:
        return _pg_moe_forward_base(st, p, lid, pos, cnt, T, *a, **k)
    saved = (_pg_MOE_LARGE_T, _pg_FINE_MIN_T)
    (_pg_MOE_LARGE_T, _pg_FINE_MIN_T) = sel
    try:
        return _pg_moe_forward_base(st, p, lid, pos, cnt, T, *a, **k)
    finally:
        (_pg_MOE_LARGE_T, _pg_FINE_MIN_T) = saved


def _pgs_routes(T, gen, dev):
    scores = torch.rand((T, 256), generator=gen, device=dev)
    z = torch.randn((256,), generator=gen, device=dev)
    u = torch.rand((256,), generator=gen, device=dev).clamp(min=1e-12)
    r = (scores + 0.03 * z / (-u.log()).sqrt()).topk(8, dim=-1)
    pairs = torch.bincount((r.indices // 32).flatten(), minlength=8)
    ids = ((r.indices - 32 * pairs.argmax()) % 256).to(torch.int32).contiguous()
    return ids, (r.values / r.values.sum(-1, keepdim=True)).float().contiguous()


def _pgs_time(fn, flush, reps=5):
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


def _pgs_select(state):
    import time
    t0 = time.monotonic()
    dev = state['post_expert_w'].device
    flush = torch.ones(64 << 20, dtype=torch.float32, device=dev)
    gen = torch.Generator(device=dev).manual_seed(20261029)
    inc = (_pg_MOE_LARGE_T, _pg_FINE_MIN_T)
    for T in (753, 3012, 16384):
        x = torch.randn((T, 4096), generator=gen, device=dev).to(torch.bfloat16)
        (ids, tw) = _pgs_routes(T, gen, dev)
        res = {}
        for sel in ((1, 1), (1, 0), (1 << 30, 1), (1 << 30, 0)):
            _PG_SEL[T] = sel
            try:
                out = _experts(x, tw, ids, state).float()
                t = _pgs_time(lambda: _experts(x, tw, ids, state), flush)
            except Exception as ex:
                print(f'PG_SKIP T={T} {sel} {type(ex).__name__}', flush=True)
                continue
            finally:
                _PG_SEL.pop(T, None)
            res[sel] = (t, out)
        cur = ((1 if T >= inc[0] else 1 << 30), (1 if inc[1] and T >= inc[1] else 0))
        if cur not in res:
            print(f'PG_KEEP T={T} incumbent untimed', flush=True)
            continue
        ref = res[cur][1]
        for (sel, (t, out)) in res.items():
            err = float((out - ref).norm() / ref.norm().clamp_min(1e-30))
            print(f'PG_CANDIDATE T={T} large={int(sel[0] == 1)} fine={sel[1]} {t:.1f}us err {err:.2e} incumbent {res[cur][0]:.1f}us', flush=True)
            if err >= 1e-2:
                res[sel] = (float('inf'), out)
        best = min(res, key=lambda s: res[s][0])
        if best != cur and res[best][0] < 0.97 * res[cur][0]:
            _PG_SEL[T] = best
            print(f'PG_SELECT T={T} large={int(best[0] == 1)} fine={best[1]} gain {res[cur][0] / res[best][0]:.4f}', flush=True)
        else:
            print(f'PG_KEEP T={T}', flush=True)
        del x, ids, tw, res
    del flush
    torch.cuda.synchronize()
    print(f'PG_SELECTOR selected={sorted(_PG_SEL)} s={time.monotonic() - t0:.1f}', flush=True)


_pgs_original_prepare = prepare


def prepare(weights):
    state = _pgs_original_prepare(weights)
    _PG_SEL.clear()
    try:
        _pgs_select(state)
    except Exception as ex:
        _PG_SEL.clear()
        print(f'PG_SELECTOR off {type(ex).__name__}: {ex}', flush=True)
    torch.cuda.synchronize()
    return state
