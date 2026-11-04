"""Appended verbatim into the self-contained check file; no timed tuning."""

_verify_original_prepare = prepare


def _verify_routes(M, gen, dev):
    # Same public distribution as the incumbent's _moe_routes, not fixed outputs
    # or fixed routing. Busiest rank is rotated into local experts 0..31.
    scores = torch.rand((M, 256), generator=gen, device=dev)
    z = torch.randn((256,), generator=gen, device=dev)
    u = torch.rand((256,), generator=gen, device=dev).clamp(min=1e-12)
    scores = scores + 0.03 * z / (-u.log()).sqrt()
    r = scores.topk(8, dim=-1)
    pairs = torch.bincount((r.indices // 32).flatten(), minlength=8)
    ids = ((r.indices - 32 * pairs.argmax()) % 256).to(torch.int32).contiguous()
    tw = (r.values / r.values.sum(-1, keepdim=True)).float().contiguous()
    return ids, tw


def _verify_close(a, b):
    # This is stricter than a generic one-ulp BF16 perturbation, intentionally:
    # a new instruction must not trade accuracy for time. Never widen to fit.
    af, bf = a.float(), b.float()
    if not bool(torch.isfinite(af).all()) or not bool(torch.isfinite(bf).all()):
        return False
    rel = (af - bf).norm() / bf.norm().clamp_min(1e-30)
    return float(rel) <= 2e-3 and torch.allclose(af, bf, rtol=2e-2, atol=2e-3)


def _verify_admit(records):
    # Every pass must reproduce >=1.5%, in both arm orders and with both
    # aggregate and paired ratios. A routing draw may not regress by >0.5%.
    import statistics
    if len(records) != 2:
        return False
    for rows in records:
        if len(rows) != 8 or {r[0] for r in rows} != set(range(8)):
            return False
        for order in (0, 1):
            ratios = [r[1 + order] for r in rows]
            paired = [r[3 + order] for r in rows]
            if not all(math.isfinite(x) and 0 < x <= 1.005 for x in ratios + paired):
                return False
            if statistics.median(ratios) > 0.985 or statistics.median(paired) > 0.985:
                return False
    return True


def _verify_graph(ext, state, scratch, x, ids, tw):
    out = torch.empty_like(x)
    args = (state["w"], scratch, x, ids, tw, out,
            state["lo"], state["cnt"], state["grid"])
    ext.okc_run(*args)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ext.okc_run(*args)
    graph.replay()
    torch.cuda.synchronize()
    return graph, out


def _verify_correctness(base, cand, state, scratches, M, gen):
    dev = state["w"][0].device
    x = torch.randn((M, HIDDEN), generator=gen, device=dev).to(torch.bfloat16)
    ids, tw = _verify_routes(M, gen, dev)
    gb, ob = _verify_graph(base, state, scratches[0], x, ids, tw)
    gc, oc = _verify_graph(cand, state, scratches[1], x, ids, tw)
    # Replay these same captured graphs across route-buffer updates. Tests
    # zero jobs and >64-route splitting as well as the N=8/16 boundary, and
    # catches stale state when switching from long jobs to short/remote ones.
    remote = torch.arange(32, 40, dtype=torch.int32, device=dev).expand(M, 8)
    for count in (None, M, 65, 64, 9, 8, 1, 0, None):
        ids.copy_(remote)
        if count is None:
            rid, rtw = _verify_routes(M, gen, dev)
            ids.copy_(rid)
            tw.copy_(rtw)
        else:
            ids[:min(M, count), 0] = 0
            tw.fill_(1 / 8)
        for _ in range(4):
            gb.replay()
            gc.replay()
            torch.cuda.synchronize()
            if not _verify_close(oc, ob):
                return False
            if count == 0 and (bool(ob.any()) or bool(oc.any())):
                return False
    return True


def _verify_time_pair(graphs, flush, events, reverse):
    import statistics
    times = [[], []]
    paired = []
    # ABBA / BAAB, 12 independent quads. Flush by READING 512 MiB; never
    # zero_/add_ it, which was shown to bias the MoE ledger's old timings.
    order = (1, 0, 0, 1) if reverse else (0, 1, 1, 0)
    e0, e1 = events
    for _ in range(12):
        quad = [[], []]
        for arm in order:
            flush.amax()
            e0.record()
            graphs[arm].replay()
            e1.record()
            e1.synchronize()
            value = float(e0.elapsed_time(e1))
            times[arm].append(value)
            quad[arm].append(value)
        paired.append(statistics.median(quad[1]) / statistics.median(quad[0]))
    return (statistics.median(times[1]) / statistics.median(times[0]),
            statistics.median(paired))


def _verify_select(state):
    import statistics
    import time
    state["verify_sync"] = {}
    if not state.get("sfx"):
        print("VERIFY_SYNC disabled: PRMT scales not exact", flush=True)
        return
    started = time.monotonic()
    base = _ext(tuple(MAIN_DEFINES) + state["sfx"])
    cand = _ext(tuple(MAIN_DEFINES) + state["sfx"] + ("-DOKC_VERIFY_SYNC=1",))
    ba, ca = tuple(base.okc_verify_resources()), tuple(cand.okc_verify_resources())
    print(f"VERIFY_SYNC resources base={ba} candidate={ca}", flush=True)
    if len(ba) != 3 or len(ca) != 3 or any(c > b for c, b in zip(ca, ba)):
        print("VERIFY_SYNC disabled: resource growth", flush=True)
        return
    dev = state["w"][0].device
    # Scratch/counters are never shared between arms or between captured
    # graphs and production. Nothing from a tuning call is a runtime output.
    scratches = (_make_scratch(dev), _make_scratch(dev))
    flush = torch.ones(128 << 20, dtype=torch.float32, device=dev)
    events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
    gen = torch.Generator(device=dev).manual_seed(2026110408)
    for M in (128, 256):
        records = []
        try:
            # Respect any incumbent preparation-time choice. This experiment
            # replaces only MAIN WGS, never the existing mma.sync fallback.
            incumbent = _MOE_SEL.get(M) or _moe_default(M, state["sfx"])
            if incumbent != tuple(MAIN_DEFINES) + state["sfx"]:
                print(f"VERIFY_SYNC M={M} keep=BASE reason=incumbent-not-WGS", flush=True)
                continue
            if not _verify_correctness(base, cand, state, scratches, M, gen):
                print(f"VERIFY_SYNC M={M} keep=BASE reason=correctness", flush=True)
                continue
            for phase in range(2):
                rows = []
                for draw in range(8):
                    if time.monotonic() - started > 75:
                        raise RuntimeError("75s tuning deadline; keep incumbent")
                    x = torch.randn((M, HIDDEN), generator=gen, device=dev).to(torch.bfloat16)
                    ids, tw = _verify_routes(M, gen, dev)
                    gb, ob = _verify_graph(base, state, scratches[0], x, ids, tw)
                    gc, oc = _verify_graph(cand, state, scratches[1], x, ids, tw)
                    if not _verify_close(oc, ob):
                        raise RuntimeError(f"draw {phase}/{draw} correctness")
                    ab, pab = _verify_time_pair((gb, gc), flush, events, False)
                    ba_ratio, pba = _verify_time_pair((gb, gc), flush, events, True)
                    rows.append((draw, ab, ba_ratio, pab, pba))
                    # Timing must not have damaged counters/output state.
                    if not _verify_close(oc, ob):
                        raise RuntimeError(f"draw {phase}/{draw} replay correctness")
                    del gb, gc
                records.append(rows)
                med = [statistics.median(r[i] for r in rows) for i in range(1, 5)]
                print(f"VERIFY_SYNC M={M} pass={phase} ratios={med} "
                      f"worst={max(max(r[1:]) for r in rows):.5f}", flush=True)
            if _verify_admit(records):
                state["verify_sync"][M] = cand
            print(f"VERIFY_SYNC M={M} selected={'SYNC8' if M in state['verify_sync'] else 'BASE'}", flush=True)
        except Exception as exc:
            # A fault in one shape must not install an unvalidated candidate.
            state["verify_sync"].pop(M, None)
            print(f"VERIFY_SYNC M={M} keep=BASE {type(exc).__name__}: {exc}", flush=True)
    torch.cuda.synchronize()


def prepare(weights):
    state = _verify_original_prepare(weights)
    state["verify_sync"] = {}
    try:
        _verify_select(state)
    except Exception as exc:
        state["verify_sync"] = {}
        print(f"VERIFY_SYNC keep=BASE {type(exc).__name__}: {exc}", flush=True)
    torch.cuda.synchronize()
    print(f"VERIFY_SYNC_FINAL selected={sorted(state['verify_sync'])}", flush=True)
    return state
