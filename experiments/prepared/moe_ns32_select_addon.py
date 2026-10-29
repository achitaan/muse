
# Skinny-ring depth selector for the weakest MoE load, decode 32 (M5; M4 skipped M=32 on budget) (lane omos/other-tracks-20261029).
# decode 32 (MAIN WGS) runs 127 us vs a 70.8 us floor and decode 16 (MID) 95 vs 52.5 us, both
# weight-streaming bound. prepare() compiles the incumbent build with one more TMA ring slot
# (-DOKC_NS=4; default 3, kernel limit 5), times it against the incumbent as cold-L2 graph
# replays over three routing draws and uses it only if >=3% faster with matching output.
# New compiles are skipped once prepare has run 200 s (300 s cap).
_MOE_SEL = {}


def _moe_default(M, sfx):
    if TINY_DEFINES and M <= TINY_MAX_M:
        return tuple(TINY_DEFINES) + sfx
    if TINY2_DEFINES and M <= TINY2_MAX_M and (sfx or "-DOKC_WGS=1" not in tuple(TINY2_DEFINES)):
        return tuple(TINY2_DEFINES) + sfx
    if MID_DEFINES and M <= MID_MAX_M:
        return tuple(MID_DEFINES) + sfx
    return tuple(_main_defines(sfx)) + sfx


def _moe_valid(M, sfx):
    import time
    inc = _moe_default(M, sfx)
    out = [inc]
    for (ns, limit) in ((4, 185.0), (5, 185.0)):
        el = time.monotonic() - _M4_T0[0]
        if el > limit:
            print(f"MOE_SKIP M={M} NS{ns} prepare budget {el:.0f}s", flush=True)
            continue
        d = inc + (f"-DOKC_NS={ns}",)
        try:
            _ext(d)
            out.append(d)
        except Exception as x:
            print(f"MOE_SKIP M={M} NS{ns} {type(x).__name__}", flush=True)
    return out


def _moe_routes(M, gen, dev):
    scores = torch.rand((M, 256), generator=gen, device=dev)
    z = torch.randn((256,), generator=gen, device=dev)
    u = torch.rand((256,), generator=gen, device=dev).clamp(min=1e-12)
    scores = scores + 0.03 * z / (-u.log()).sqrt()
    r = scores.topk(8, dim=-1)
    pairs = torch.bincount((r.indices // 32).flatten(), minlength=256 // 32)
    ids = ((r.indices - 32 * pairs.argmax()) % 256).to(torch.int32).contiguous()
    return ids, (r.values / r.values.sum(-1, keepdim=True)).float().contiguous()


def _moe_select(state):
    import time
    t0 = time.monotonic()
    sfx = state.get("sfx", ())
    dev = state["w"][0].device
    flush = torch.ones(64 << 20, dtype=torch.float32, device=dev)
    gen = torch.Generator(device=dev).manual_seed(20261029)
    for M in (32,):
        cands = _moe_valid(M, sfx)
        inc = _moe_default(M, sfx)
        if inc not in cands or len(cands) < 2:
            continue
        draws = []
        for _ in range(3):
            x = torch.randn((M, HIDDEN), generator=gen, device=dev).to(torch.bfloat16)
            ids, tw = _moe_routes(M, gen, dev)
            draws.append((x, ids, tw))
        total, outs = {}, {}
        for d in cands:
            ext = _ext(d)
            acc = 0.0
            for (k, (x, ids, tw)) in enumerate(draws):
                out = torch.empty((M, HIDDEN), dtype=torch.bfloat16, device=dev)
                run = lambda: ext.okc_run(state["w"], state["scratch"], x, ids, tw, out, state["lo"], state["cnt"], state["grid"])
                run()
                torch.cuda.synchronize()
                outs[d, k] = out.float().clone()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    run()
                t = []
                for _ in range(9):
                    flush.sum()
                    (e0, e1) = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                    e0.record()
                    g.replay()
                    e1.record()
                    e1.synchronize()
                    t.append(e0.elapsed_time(e1) * 1000.0)
                del g
                acc += sorted(t)[4]
            total[d] = acc
        for d in cands:
            err = max(float((outs[d, k] - outs[inc, k]).norm() / outs[inc, k].norm().clamp_min(1e-30)) for k in range(3))
            name = [n for (n, v) in (("TINY", TINY_DEFINES), ("TINY2", TINY2_DEFINES), ("MID", MID_DEFINES)) if v and tuple(v) + sfx == d[:len(tuple(v) + sfx)]]
            name = [(name[0] if name else "MAIN") + ("+NS4" if "-DOKC_NS=4" in d else "+NS5" if "-DOKC_NS=5" in d else "")]
            print(f"MOE_CANDIDATE M={M} {name[0] if name else 'MAIN'} sum3={total[d]:.2f} err={err:.2e} incumbent={total[inc]:.2f}", flush=True)
            total[d] = total[d] if err < 1e-2 else float("inf")
        best = min(cands, key=lambda d: total[d])
        if best != inc and total[best] < 0.97 * total[inc]:
            _MOE_SEL[M] = best
            print(f"MOE_SELECT M={M} gain={total[inc] / total[best]:.4f}", flush=True)
        else:
            print(f"MOE_KEEP M={M}", flush=True)
    del flush
    torch.cuda.synchronize()
    print(f"MOE_SELECTOR selected={sorted(_MOE_SEL)} s={time.monotonic() - t0:.1f}", flush=True)


def kernel_fn(data, state):
    """Run one MoE layer. Fresh output every call; no host syncs; the only
    per-call work is picking the compiled variant by token count and
    launching it in CHUNK_TOKENS-token slices."""
    cfg = data["config"]
    x = data["hidden_states"].contiguous()
    M = x.shape[0]
    assert cfg["local_start"] == state["lo"] and cfg["local_count"] == state["cnt"]
    ids = data["topk_ids"].to(torch.int32).contiguous()
    if ids.data_ptr() & 15:  # the WGS routing loads topk_ids 16 B at a time (an offset view gets an aligned copy)
        ids = ids.clone()
    tw = data["topk_weights"].to(torch.float32).contiguous()
    out = torch.empty((M, HIDDEN), dtype=torch.bfloat16, device=x.device)
    sfx = state.get("sfx", ())
    ext = _ext(_MOE_SEL.get(M) or _moe_default(M, sfx))
    for m0 in range(0, M, CHUNK_TOKENS):
        m1 = min(M, m0 + CHUNK_TOKENS)
        ext.okc_run(state["w"], state["scratch"], x[m0:m1], ids[m0:m1], tw[m0:m1], out[m0:m1],
                    state["lo"], state["cnt"], state["grid"])
    return out


_moe_sel_original_prepare = prepare
_M4_T0 = [0.0]


def prepare(weights):
    import time
    _M4_T0[0] = time.monotonic()
    state = _moe_sel_original_prepare(weights)
    _MOE_SEL.clear()
    try:
        _moe_select(state)
    except Exception as x:
        _MOE_SEL.clear()
        print(f"MOE_SELECTOR off {type(x).__name__}: {x}", flush=True)
    torch.cuda.synchronize()
    return state
