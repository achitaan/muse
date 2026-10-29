
# First-call split autotune per shape (lane omos/decode-fresh-20261029). The first eager call of each
# launch shape (never under graph capture) times split counts around the incumbent plan, and the
# cooperative-finalize toggle where all CTAs stay resident, as cold-L2 CUDA-graph replays interleaved
# with the incumbent. A variant is used only if it is >=3% faster than the better of two incumbent
# timings and its output matches the incumbent's (rel-L2 < 1e-2); otherwise the incumbent plan runs.
_AT_SEL = {}
_AT_DONE = set()
_AT_FLUSH = {}


def _at_key(data):
    cfg = data["config"]
    return (cfg["batch"], cfg["q_len"], int(cfg["window"]), cfg["num_kv_heads"] // 2,
            data["page_table"].shape[1], cfg["page_size"])


def _at_variants(data):
    cfg = data["config"]
    (B, L, window, P) = (cfg["batch"], cfg["q_len"], cfg["window"], cfg["page_size"])
    pairs = cfg["num_kv_heads"] // 2
    max_pages = data["page_table"].shape[1]
    device = data["qkv"].device
    sms = _num_sms(device)
    tuned = _TUNED.get((B, L, bool(window), pairs)) or {}
    plan = dict(_plan(B, pairs, L, window, max_pages, P, device), **tuned)
    if plan.get("cs", 0) > 1:
        return []
    keys_max = min(window + L - 1, max_pages * P) if window else max_pages * P
    tiles = max(1, (keys_max + TILE_KEYS - 1) // TILE_KEYS)
    flat = bool(plan["flat"])
    cap = B * tiles if flat else tiles
    n = plan["n_splits"]
    out = []
    for f in (0.5, 0.75, 1.25, 1.5, 2.0):
        v = max(1, min(cap, int(round(n * f))))
        if v == n or any(o["n_splits"] == v for o in out):
            continue
        ov = dict(tuned, n_splits=v)
        if "nc" in ov:
            ov["nc"] = max(1, min(ov["nc"], v))
        if ov.get("coop") == 1 and (v * pairs if flat else v * B * pairs) > 2 * sms:
            ov["coop"] = 0
        out.append(ov)
    if not flat:
        coop = int(plan.get("coop", 0 if n < 64 else int(n * B * pairs <= 2 * sms)))
        if coop == 1 or n * B * pairs <= 2 * sms:
            out.append(dict(tuned, n_splits=n, coop=1 - coop))
    return out


def _at_time(fn, flush, reps=9):
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


def _at_tune(data, key):
    device = data["qkv"].device
    flush = _AT_FLUSH.get(str(device))
    if flush is None:
        flush = _AT_FLUSH[str(device)] = torch.ones(64 << 20, dtype=torch.float32, device=device)
    ref = _launch(data).float()
    base = _at_time(lambda: _launch(data), flush)
    res = []
    for ov in _at_variants(data):
        try:
            y = _launch(data, **ov).float()
            err = float((y - ref).norm() / ref.norm().clamp_min(1e-30))
            t = _at_time(lambda: _launch(data, **ov), flush)
        except Exception as ex:
            print(f"AT_SKIP {key} {ov} {type(ex).__name__}", flush=True)
            continue
        res.append((t, err, ov))
    base = min(base, _at_time(lambda: _launch(data), flush))
    torch.cuda.synchronize()
    ok = [r for r in res if r[1] < 1e-2]
    best = min(ok, key=lambda r: r[0]) if ok else None
    picked = best is not None and best[0] < 0.97 * base
    if picked:
        _AT_SEL[key] = best[2]
    print(f"AT {key} inc {base:.2f} " + " ".join(f"{r[2]['n_splits']}/{r[2].get('coop', '-')}:{r[0]:.2f}" + ("" if r[1] < 1e-2 else "!") for r in res)
          + (f" SELECT {best[2]} {base / best[0]:.4f}" if picked else " KEEP"), flush=True)


_at_prev_kernel_fn = kernel_fn


def kernel_fn(data: Inputs) -> torch.Tensor:
    if not _supported(data["config"]) or not _fits(data):
        return _fallback(data)
    key = _at_key(data)
    if key not in _AT_DONE and not torch.cuda.is_current_stream_capturing():
        _AT_DONE.add(key)
        try:
            _at_tune(data, key)
        except Exception as ex:
            _AT_SEL.pop(key, None)
            print(f"AT_OFF {key} {type(ex).__name__}: {ex}", flush=True)
    ov = _AT_SEL.get(key)
    return _launch(data, **ov) if ov else _launch(data)
