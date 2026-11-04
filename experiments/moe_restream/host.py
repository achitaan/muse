# TQ64 experimental append. The incumbent source/module/state are kept intact.
_RS_EXT = None
_rs_base_prepare = _moe_sel_original_prepare  # pack/build only; legacy selector must not run before TQ64
_rs_base_kernel_fn = kernel_fn
_rs_torch_load_inline = load_inline
_RS_GATE_SECONDS = 285.0
_RS_POST_SELECT_SECONDS = 290.0


def _rs_cxx20_flags(flags):
    # One module-local compiler policy for BOTH the original _ext and TQ64.
    # Do not modify torch.utils.cpp_extension or any external loaded module.
    result, skip = [], False
    for flag in flags or ():
        if skip:
            skip = False
        elif flag in ("-std", "--std"):
            skip = True
        elif not flag.startswith(("-std=", "--std=")):
            result.append(flag)
    return result + ["-std=c++20"]


def load_inline(*args, **kwargs):
    kwargs["extra_cflags"] = _rs_cxx20_flags(kwargs.get("extra_cflags"))
    kwargs["extra_cuda_cflags"] = _rs_cxx20_flags(kwargs.get("extra_cuda_cflags"))
    assert kwargs["extra_cflags"][-1] == kwargs["extra_cuda_cflags"][-1] == "-std=c++20"
    try:
        print(f"MOE_RESTREAM_DIALECT module={kwargs.get('name', '?')} host_std=c++20 cuda_std=c++20 "
              "policy=shared_incumbent_and_TQ64", flush=True)
    except Exception:
        pass
    return _rs_torch_load_inline(*args, **kwargs)


def _rs_status(state, start, stage, reason=None):
    sfx = tuple(state.get("sfx", ()))
    defines = tuple(_MOE_SEL.get(256) or _moe_default(256, sfx))
    layout = "wgs" if "-DOKC_WGS=1" in defines else "unit" if "-DOKC_TINY=1" in defines else "unit+fat"
    prmt = "-DOKC_PRMTSC=1" in sfx
    elapsed = _mh_time.monotonic() - start
    remaining = 300.0 - elapsed
    gate_remaining = _RS_GATE_SECONDS - elapsed
    prefix = "MOE_RESTREAM_SKIP" if reason else "MOE_RESTREAM_PREP"
    print(f"{prefix} stage={stage}" + (f" reason={reason}" if reason else "") +
          f" layout={layout} scale_decode={'prmt' if prmt else 'lut'} prmt={int(prmt)} "
          f"elapsed_s={elapsed:.2f} remaining_s={remaining:.2f} gate_remaining_s={gate_remaining:.2f}", flush=True)
    return prmt, gate_remaining


def _rs_admit(state, start):
    prmt, remaining = _rs_status(state, start, "pre_build")
    if not prmt:
        _rs_status(state, start, "pre_build", "nonPRMT")
        return False
    if remaining < 60.0:
        _rs_status(state, start, "pre_build", "insufficient_compile_budget")
        return False
    return True


def _rs_ext(sfx):
    global _RS_EXT
    if _RS_EXT is None:
        defs = tuple(MAIN_DEFINES) + tuple(sfx) + ("-DOKC_DQ7=1", "-DOKC_NS=3", "-DOKC_NSF=3")
        _RS_EXT = load_inline(
            name="okc_moe_tq64_k128_v2_sm90a", cpp_sources=CPP_SRC + "\nstd::vector<int64_t> rs_attrs();\n",
            cuda_sources=_RS_CUDA, functions=["okc_run", "okc_set_pdl", "rs_attrs"],
            extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "-lineinfo", *defs],
            verbose=False,
        )
        _RS_EXT.okc_set_pdl(0)
    return _RS_EXT


def _rs_pack(codes, sf, rows):
    # Lossless permutation: [256-row block][K128 stage][q64][4608 bytes].
    nb, nub = rows.numel() // 16, rows.numel() // 256
    w = _pack_codes_nat(codes, rows).view(nb, 8, 8, 8, 4, 4)
    w = w.permute(0, 1, 2, 5, 3, 4).contiguous().view(nb, 256, 32)
    w = w.view(nub, 4, 4, 32, 2, 4, 32).permute(0, 3, 1, 4, 2, 6, 5).contiguous()
    cb = w.view(torch.uint8).view(nub, 32, 4, 4096)
    s = sf.view(torch.uint8).index_select(0, rows).view(nub, 4, 4, 2, 8, 32, 2, 4)
    sb = s.permute(0, 5, 1, 6, 2, 4, 7, 3).contiguous().view(nub, 32, 4, 512)
    return torch.cat((cb, sb), dim=3).contiguous().view(-1)


def _rs_run(data, state):
    x = data["hidden_states"].contiguous()
    ids = data["topk_ids"].to(torch.int32).contiguous()
    if ids.data_ptr() & 15:
        ids = ids.clone()
    tw = data["topk_weights"].to(torch.float32).contiguous()
    cfg = data["config"]
    assert cfg["local_start"] == state["lo"] and cfg["local_count"] == state["cnt"]
    out = torch.empty((x.shape[0], HIDDEN), dtype=torch.bfloat16, device=x.device)
    state["_rs_ext"].okc_run(state["_rs_w"], state["_rs_scratch"], x, ids, tw, out,
                              state["lo"], state["cnt"], state["grid"])
    return out


def kernel_fn(data, state):
    if data["hidden_states"].shape[0] in state.get("_rs_selected", ()):
        return _rs_run(data, state)
    return _rs_base_kernel_fn(data, state)


def _rs_check(ref, out):
    a, b = ref.float(), out.float()
    if not bool(torch.isfinite(a).all() & torch.isfinite(b).all()):
        raise ValueError("TQ64 nonfinite output")
    diff = b - a
    scale = a.norm(dim=-1)
    row = (diff.norm(dim=-1) / scale.clamp_min(0.1 * scale.mean()).clamp_min(1e-20)).max()
    err = diff.norm() / a.norm().clamp_min(1e-20)
    if float(err) > .005 or float(row) > .02:
        raise ValueError(f"TQ64 numerical gate l2={float(err):.5g} row={float(row):.5g}")
    return float(err), float(row)


def _rs_gate(weights, state, start):
    deadline = start + _RS_GATE_SECONDS
    if not _rs_admit(state, start):
        return
    dev = state["w"][0].device
    if torch.cuda.get_device_capability(dev)[0] != 9:
        _rs_status(state, start, "pre_build", "unsupported_arch")
        return
    module = _rs_ext(state["sfx"])
    attrs = list(module.rs_attrs())
    _rs_status(state, start, "post_compile")
    print(f"MOE_RESTREAM_RES attrs_regs_local_smem_struct_threads={attrs}", flush=True)
    if attrs[2] > 227328 or attrs[4] < 576:
        raise ValueError("TQ64 resource limit")
    if _mh_time.monotonic() > deadline - 20:
        _rs_status(state, start, "post_compile", "insufficient_gate_budget")
        return
    rb, g = torch.arange(HIDDEN // 8, device=dev), torch.arange(8, device=dev)
    rowsA = torch.stack((8 * rb[:, None] + g[None, :], HIDDEN + 8 * rb[:, None] + g[None, :]), 1).reshape(-1)
    rowsB = torch.arange(HIDDEN, device=dev)
    lo, cnt = state["lo"], state["cnt"]
    wa = torch.cat([_rs_pack(weights["w13"][e], weights["w13_sf"][e], rowsA) for e in range(lo, lo + cnt)])
    wb = torch.cat([_rs_pack(weights["w2"][e], weights["w2_sf"][e], rowsB) for e in range(lo, lo + cnt)])
    private_w = list(state["w"])
    private_w[12], private_w[13] = wa, wb
    state["_rs_ext"], state["_rs_w"], state["_rs_scratch"] = module, private_w, _make_scratch(dev)
    torch.cuda.synchronize(dev)
    _rs_status(state, start, "post_pack")
    if _mh_time.monotonic() > deadline - 10:
        _rs_status(state, start, "post_pack", "insufficient_gate_budget")
        return
    flush = torch.empty(128 << 20, dtype=torch.float32, device=dev).fill_(1.)
    flush.amax()
    torch.cuda.synchronize(dev)
    gen = torch.Generator(device=dev).manual_seed(20261105)

    def data_for(ids):
        return {"hidden_states": torch.randn((256, HIDDEN), generator=gen, device=dev).to(torch.bfloat16),
                "topk_ids": ids, "topk_weights": torch.full((256, 8), .125, device=dev),
                "config": {"local_start": lo, "local_count": cnt}}

    # Stress all four token WGs, then an empty replay. No reference call is
    # made on all-remote inputs because the unmodified base has a known J0 bug.
    hot = data_for(torch.arange(lo, lo + 8, device=dev, dtype=torch.int32).repeat(256, 1))
    _rs_check(_rs_base_kernel_fn(hot, state), _rs_run(hot, state))
    empty = data_for(((torch.arange(8, device=dev, dtype=torch.int32) + lo + cnt) % 256).repeat(256, 1))
    zgraph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(zgraph):
        zout = _rs_run(empty, state)
    for _ in range(3):
        zgraph.replay()
    torch.cuda.synchronize(dev)
    if bool((zout != 0).any()):
        raise ValueError("TQ64 nonzero empty replay")
    _rs_check(_rs_base_kernel_fn(hot, state), _rs_run(hot, state))
    print(f"MOE_RESTREAM_CHECK hot_empty=pass elapsed_s={_mh_time.monotonic()-start:.2f} "
          f"gate_remaining_s={deadline-_mh_time.monotonic():.2f}", flush=True)
    del hot, empty, zgraph, zout

    def time_one(graph):
        flush.amax()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); graph.replay(); b.record(); b.synchronize()
        return a.elapsed_time(b) * 1000

    passed = True
    all_rows = []
    for pass_id in range(2):
        rows = []
        for draw in range(16):
            if _mh_time.monotonic() > deadline - 3:
                raise TimeoutError("TQ64 gate deadline")
            x = torch.randn((256, HIDDEN), generator=gen, device=dev).to(torch.bfloat16)
            ids, tw = _moe_routes(256, gen, dev)
            data = {"hidden_states": x, "topk_ids": ids, "topk_weights": tw,
                    "config": {"local_start": lo, "local_count": cnt}}
            local = ids[(ids >= lo) & (ids < lo + cnt)] - lo
            counts = torch.bincount(local.long(), minlength=cnt).cpu().tolist()
            stats = _mh_stats(counts, MAIN_DEFINES)
            before, out = _rs_base_kernel_fn(data, state), _rs_run(data, state)
            err, rowerr = _rs_check(before, out)
            torch.cuda.synchronize(dev)
            ga, gb = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
            with torch.cuda.graph(ga):
                before = _rs_base_kernel_fn(data, state)
            with torch.cuda.graph(gb):
                out = _rs_run(data, state)
            t = _mh_abba(time_one, ga, gb, deadline, rounds=6)
            _rs_check(before, out)
            ratio = 1 / t["ratio"]
            rows.append((ratio, stats["dup_jobs"]))
            print(f"MOE_RESTREAM_DRAW M=256 pass={pass_id} draw={draw} D={stats['D']} R={stats['R']} "
                  f"dup={stats['dup_jobs']} bytes_saved={stats['dup_jobs']*_MH_EXPERT_BULK} "
                  f"base_us={t['inc_us']:.3f} tq_us={t['stream_us']:.3f} tq_over_base={ratio:.5f} "
                  f"err={err:.3g} row={rowerr:.3g}", flush=True)
            del ga, gb, before, out, data, x, ids, tw, local
        med = _mh_statistics.median(r for r, _ in rows)
        clean = [r for r, dup in rows if not dup]
        clean_med = _mh_statistics.median(clean) if clean else None
        ok = med <= .97 and (clean_med is not None and clean_med <= 1.01)
        passed &= ok
        all_rows.extend(rows)
        print(f"MOE_RESTREAM_PASS M=256 pass={pass_id} median={med:.5f} nodup_median={clean_med} pass={ok}", flush=True)
    if passed:
        state["_rs_selected"].add(256)
    print(f"MOE_RESTREAM_SELECT shapes={sorted(state['_rs_selected'])} "
          f"median={_mh_statistics.median(r for r,_ in all_rows):.5f} elapsed_s={_mh_time.monotonic()-start:.2f}", flush=True)


def _rs_is_cached(defines):
    key = tuple(defines)
    if DQ7 and "-DOKC_DQ7=1" not in key:
        key += ("-DOKC_DQ7=1",)
    return key in _EXTS


def _rs_post_select(state, start):
    # Preserve the small-shape selector's 3 draws / 9 replays / 3% criterion,
    # but only AFTER TQ64 and only with already-compiled modules. No lazy
    # fallback compile can consume the candidate's reserved build/gate time.
    # M128/M256 remain default: both supplied runs selected no override there.
    # In particular the M256 comparator cannot change after the TQ64 gate.
    deadline = start + _RS_POST_SELECT_SECONDS
    if _mh_time.monotonic() > deadline - 2:
        print("MOE_SELECTOR_SKIP reason=post_TQ_budget policy=cached_small_only", flush=True)
        return
    print("MOE_SELECTOR_POLICY phase=after_TQ64 shapes=16,32 cached_only=1 skipped_shapes=128,256", flush=True)
    dev, sfx = state["w"][0].device, tuple(state.get("sfx", ()))
    flush = torch.empty(64 << 20, dtype=torch.float32, device=dev).fill_(1.)
    flush.sum()
    gen = torch.Generator(device=dev).manual_seed(20261029)
    for M in (16, 32):
        try:
            inc = _moe_default(M, sfx)
            cands = [d for d in _moe_valid(M, sfx) if _rs_is_cached(d)]
            if inc not in cands or len(cands) < 2:
                continue
            draws = []
            for _ in range(3):
                if _mh_time.monotonic() >= deadline:
                    raise TimeoutError("small-selector setup budget")
                x = torch.randn((M, HIDDEN), generator=gen, device=dev).to(torch.bfloat16)
                ids, tw = _moe_routes(M, gen, dev)
                draws.append((x, ids, tw))
            total, outs = {}, {}
            for d in cands:
                ext = _ext(d)  # cache hit proved above; cannot start a compile
                acc = 0.
                for k, (x, ids, tw) in enumerate(draws):
                    if _mh_time.monotonic() >= deadline:
                        raise TimeoutError("small-selector capture budget")
                    out = torch.empty((M, HIDDEN), dtype=torch.bfloat16, device=dev)
                    run = lambda: ext.okc_run(state["w"], state["scratch"], x, ids, tw, out, state["lo"], state["cnt"], state["grid"])
                    run()
                    torch.cuda.synchronize(dev)
                    outs[d, k] = out.float().clone()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        run()
                    times = []
                    for _ in range(9):
                        if _mh_time.monotonic() >= deadline:
                            raise TimeoutError("small-selector replay budget")
                        flush.sum()
                        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        a.record(); graph.replay(); b.record(); b.synchronize()
                        times.append(a.elapsed_time(b) * 1000)
                    acc += sorted(times)[4]
                    del graph
                total[d] = acc
            for d in cands:
                err = max(float((outs[d, k] - outs[inc, k]).norm() / outs[inc, k].norm().clamp_min(1e-30)) for k in range(3))
                if not (err < .01):
                    total[d] = float("inf")
            best = min(cands, key=lambda d: total[d])
            if best != inc and total[best] < .97 * total[inc]:
                _MOE_SEL[M] = best
            print(f"MOE_SELECTOR_CACHED M={M} candidates={len(cands)} selected_override={M in _MOE_SEL} "
                  f"elapsed_s={_mh_time.monotonic()-start:.2f}", flush=True)
        except Exception as exc:
            # Never adopt incomplete measurements, or undo an accepted TQ64
            # result because this optional legacy step ran out of time.
            print(f"MOE_SELECTOR_SKIP M={M} reason={type(exc).__name__} detail={str(exc)[-300:]}", flush=True)
            break


def prepare(weights):
    start = _mh_time.monotonic()
    _MOE_SEL.clear()
    state = _rs_base_prepare(weights)
    state["_rs_selected"] = set()
    try:
        _rs_status(state, start, "core_ready")
        _rs_gate(weights, state, start)
    except Exception as exc:
        state["_rs_selected"].clear()
        try:
            _rs_status(state, start, "gate_exception", "diagnostic_error")
            print(f"MOE_RESTREAM_ERROR type={type(exc).__name__} detail={str(exc).replace(chr(10), ' ')[-1800:]}", flush=True)
        except Exception:
            pass
    if not state["_rs_selected"]:
        for key in ("_rs_ext", "_rs_w", "_rs_scratch"):
            state.pop(key, None)
    try:
        _rs_post_select(state, start)
        _rs_status(state, start, "done")
        print(f"MOE_RESTREAM_READY shapes={sorted(state['_rs_selected'])}", flush=True)
    except Exception:
        pass
    return state
