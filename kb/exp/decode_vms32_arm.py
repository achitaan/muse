# VMS32 integration and honest same-input reference gate. Prepare-only diagnostics
# never retain inputs/outputs in production state. No runtime data-dependent pick.
import math as _dv_math
import statistics as _dv_stats
import time as _dv_time

_DV_PREPARE, _DV_KERNEL, _DV_MOE = prepare, kernel_fn, _moe
DV_PASSES, DV_ROUNDS = 3, 16
DV_NULL = (0.995, 1.005)
DV_RATIO, DV_ERROR_RATIO = 0.995, 1.10
DV_BUILD_END, DV_GATE_END = 210.0, 270.0


def _dv_deadline(deadline):
    if _dv_time.monotonic() >= deadline:
        raise TimeoutError("VMS32 prepare diagnostic deadline")


class _DVAdapter:
    def __init__(self, ext, vs):
        self.ext, self.vs = ext, vs

    def __deepcopy__(self, memo):
        # Like the incumbent _RouterExt, compiled modules are shared handles;
        # unlike it, this adapter owns scratch, which must follow state copies.
        import copy
        result = type(self)(self.ext, copy.deepcopy(self.vs, memo))
        memo[id(self)] = result
        return result

    def okc_set_pdl(self, on):
        pass  # VMS stage A is ordered after routing; B/finalizers use their own PDL chain.

    def okc_run(self, w, scratch, x, ids, tw, out, lo, nloc, grid):
        if x.shape != (32, 4096) or lo != 0 or nloc != 32:
            raise RuntimeError("VMS32 requires 32 tokens and local ids [-1,31]")
        if x.data_ptr() & 15 or not x.is_contiguous():
            x = x.contiguous().clone()
        values = dict(self.vs, x=x, ids=ids, tw=tw, out=out, mode=7, cfg=0)
        self.ext.vms_run(*[values[n] for n in VMSGATE_ARGS])


class _DVObserver:
    def __init__(self, ext, obs):
        self.ext, self.obs = ext, obs

    def okc_set_pdl(self, on):
        return self.ext.okc_set_pdl(on)

    def okc_run(self, w, scratch, x, ids, tw, out, lo, nloc, grid):
        self.obs.update(x=x, ids=ids, tw=tw, moe=out, candidate=isinstance(self.ext, _DVAdapter))
        return self.ext.okc_run(w, scratch, x, ids, tw, out, lo, nloc, grid)


def _moe(M=None, state=None):
    ext = state.get("_dv_adapter") if state is not None and M == 32 else None
    if ext is None:
        ext = _DV_MOE(M, state)
    obs = state.get("_dv_observe") if state is not None else None
    return _DVObserver(ext, obs) if obs is not None else ext


def kernel_fn(data, state):
    cfg = data["config"]
    key = (cfg["batch"], cfg["q_len"], bool(cfg["window"]))
    adapter = state.get("dvms32_pick", {}).get(key)
    if adapter is not None and cfg["tokens"] == 32:
        return _DV_KERNEL(data, dict(state, _dv_adapter=adapter))
    return _DV_KERNEL(data, state)


def _dv_capture(data, st, adapter=None):
    ss = _vd_state(st)
    ss["_dv_observe"] = {}
    ss["_dv_adapter"] = adapter
    stream = torch.cuda.Stream(device=st["qkv"].device)
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            _DV_KERNEL(data, ss)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = _DV_KERNEL(data, ss)
    graph.replay()
    torch.cuda.synchronize()
    return dict(graph=graph, output=output, obs=ss["_dv_observe"], state=ss)


def _dv_error(a, ref):
    # checks.py: flatten all leading dimensions into rows; row floor is 0.1
    # times the MEAN row norm (not the RMS). NaN/Inf always veto admission.
    if a.shape != ref.shape:
        return float("inf"), float("inf")
    a, ref = a.float().reshape(-1, a.shape[-1]), ref.float().reshape(-1, ref.shape[-1])
    if not bool(torch.isfinite(a).all() & torch.isfinite(ref).all()):
        return float("inf"), float("inf")
    d, n = (a - ref).norm(dim=-1), ref.norm(dim=-1)
    return (float(d.norm() / n.norm().clamp_min(1e-12)),
            float((d / torch.maximum(n, (0.1 * n.mean()).clamp_min(1e-12))).max()))


def _dv_numeric(new, inc, tol):
    return (all(_dv_math.isfinite(v) and v >= 0 for v in (*new, *inc, tol))
            and new[0] <= tol and new[1] <= 5 * tol
            and inc[0] <= tol and inc[1] <= 5 * tol
            and new[0] <= DV_ERROR_RATIO * inc[0])


def _dv_admit(rows):
    return (len(rows) == DV_PASSES and all(
        r["numeric"] and DV_NULL[0] <= r["null"] <= DV_NULL[1]
        and DV_NULL[0] <= r["null_ratio"] <= DV_NULL[1]
        and _dv_math.isfinite(r["paired"]) and 0 < r["paired"] < 1.000
        for r in rows) and _dv_stats.median(r["paired"] for r in rows) <= DV_RATIO)


def _dv_rms(x, eps=1e-5):
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)


def _dv_coeff(gate):
    w = gate.float() / 0.3
    beta = torch.sigmoid(w)
    return (torch.sigmoid(-w) * (1 + beta)).clamp_min(1e-3).sqrt(), beta


def _dv_ref_moe(x, ids, tw, weights, cache, deadline):
    # Independent E2M1/E4M3 dequant + FP32 GEMMs; public MoE equations.
    # ids are LOCAL expert indices, -1 remote, just as the unchanged route emits.
    out = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
    for e in torch.unique(ids).tolist():
        if not 0 <= e < 32:
            continue
        _dv_deadline(deadline)
        if e not in cache:
            cache[e] = (_vmsgate_dequant(weights["w13"][e], weights["w13_sf"][e], weights["w13_gs"][e]),
                        _vmsgate_dequant(weights["w2"][e], weights["w2_sf"][e], weights["w2_gs"][e]))
        w13, w2 = cache[e]
        tok, slot = (ids == e).nonzero(as_tuple=True)
        h = x[tok].float() @ w13.T
        inter = torch.nn.functional.silu(h[:, :4096]) * h[:, 4096:]
        y = (inter @ w2.T) * (1.0 + weights["post_expert_norm"].float())
        out.index_add_(0, tok, _dv_rms(y, 1e-8) * tw[tok, slot, None])
    return out


def _dv_reference(data, weights, ids, cache, deadline, fp8=False):
    # Standalone FP32 public layer equations (no imports/calls into judge code).
    # Compute the reference with the SAME expert ids as the incumbent/candidate,
    # matching the judge's ref_back contract. Never round layer activations to BF16.
    cfg, dev = data["config"], data["attn_input"].device
    _dv_deadline(deadline)
    qkv = data["attn_input"].float() @ weights["qkv_gate"].float().T
    q, k, v, gate = qkv.split([1024, 128, 128, 1024], dim=-1)
    q, k = _dv_rms(q.reshape(32, 16, 64)), _dv_rms(k.reshape(32, 2, 64))
    if cfg["rope_theta"]:
        freq = 1.0 / (cfg["rope_theta"] ** (torch.arange(0, 64, 2, device=dev).float() / 64))
        ang = data["positions"].float()[:, None] * freq
        c, s = ang.cos()[:, None], ang.sin()[:, None]
        def rotate(x):
            a, b = x[..., 0::2], x[..., 1::2]
            return torch.stack((a * c - b * s, a * s + b * c), -1).flatten(-2)
        q, k = rotate(q), rotate(k)
    if fp8:
        q = q.to(torch.float8_e4m3fn).float() * cfg["qk_scale_factor"]
    else:
        q = (q * cfg["qk_scale_factor"]).to(torch.bfloat16).float()
    def quant(x, scale):
        return (x / scale.float()).clamp(-448, 448).to(torch.float8_e4m3fn)
    kn = quant(k, data["k_scale"])
    vn = quant(v.reshape(32, 2, 64), data["v_scale"])
    out = torch.empty((32, 16, 64), device=dev)
    for b in range(32):
        _dv_deadline(deadline)
        seq = int(data["seq_lens"][b])
        first = max(0, seq - cfg["window"] + 1) if cfg["window"] else 0
        past = torch.arange(first, seq, device=dev)
        rows = data["page_table"][b, past // cfg["page_size"]].long() * cfg["page_size"] + past % cfg["page_size"]
        keys = torch.cat((data["k_cache"].view(-1, 2, 64)[rows].float(), kn[b:b + 1].float())) * data["k_scale"].float()
        vals = torch.cat((data["v_cache"].view(-1, 2, 64)[rows].float(), vn[b:b + 1].float())) * data["v_scale"].float()
        att = torch.einsum("kgd,tkd->kgt", q[b].reshape(2, 8, 64), keys) * cfg["sm_scale"]
        out[b] = torch.einsum("kgt,tkd->kgd", att.softmax(-1), vals).reshape(16, 64)
    o = _dv_rms(out).reshape(32, 1024) * torch.sigmoid(gate)
    a = o @ weights["o"].float().T
    alpha1, beta1 = _dv_coeff(weights["attn_residual_gate"])
    stream1 = alpha1 * data["stream"].float() + beta1 * _dv_rms(a, 1e-8)
    h = _dv_rms(stream1) * (1.0 + weights["pre_ffn_norm"].float())
    scores = torch.sigmoid((h.double() @ weights["router"].double().T) * ROUTER_MULT).float()
    picked = scores.gather(1, ids.long())
    tw = picked / (picked.sum(-1, keepdim=True) + 1e-15)
    p = _dv_rms(h @ weights["pooler"].float().T) * (1.0 + weights["pre_expert_norm"].float())
    local = (ids.long() - data["local_start"].long()) % 256
    moe = _dv_ref_moe(p, local, tw, weights, cache, deadline)
    f = (moe @ weights["unpooler"].float().T) * (1.0 + weights["post_ffn_norm"].float())
    alpha2, beta2 = _dv_coeff(weights["ffn_residual_gate"])
    stream2 = alpha2 * stream1 + beta2 * _dv_rms(f, 1e-8)
    hidden = _dv_rms(stream2) * (1.0 + weights["next_pre_attn_norm"].float())
    return dict(hidden=hidden.to(torch.bfloat16), update=stream2 - alpha1 * alpha2 * data["stream"].float(),
                k=kn, v=vn)


def _dv_quad(base, other, data, snap, before, flush, sink, order, deadline):
    times = [[], []]
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for i in order:
        _dv_deadline(deadline)
        _vd_restore(data, snap, before)
        torch.sum(flush, dim=0, out=sink)  # READ512; outside timed events
        a.record()
        (other if i else base)["graph"].replay()
        b.record()
        b.synchronize()
        us = a.elapsed_time(b) * 1000.0
        if not _dv_math.isfinite(us) or us <= 0:
            raise RuntimeError("nonpositive/nonfinite CUDA interval")
        times[i].append(us)
    return times


def _dv_measure(base, null, new, data, snap, before, flush, sink, deadline):
    ratios, nulls, old_us, new_us, na, nb = [], [], [], [], [], []
    for turn in range(DV_ROUNDS):
        # Every round has BOTH ABBA and BAAB, for the arm and a distinct
        # same-code/same-input base/base capture. Reverse control placement.
        for kind in (("new", "null") if turn % 2 == 0 else ("null", "new")):
            for order in ((0, 1, 1, 0), (1, 0, 0, 1)):
                a, b = _dv_quad(base, new if kind == "new" else null, data, snap, before, flush, sink, order, deadline)
                if kind == "new":
                    ratios.append(sum(b) / sum(a))
                    old_us.extend(a)
                    new_us.extend(b)
                else:
                    nulls.append(sum(b) / sum(a))
                    na.extend(a)
                    nb.extend(b)
    med = _dv_stats.median
    return dict(old_us=med(old_us), new_us=med(new_us), ratio=med(new_us) / med(old_us),
                paired=med(ratios), null=med(nulls), null_ratio=med(nb) / med(na))


def _dv_replay(run, data, snap, before):
    _vd_restore(data, snap, before)
    run["graph"].replay()
    torch.cuda.synchronize()


def _dv_observations(run):
    return dict(output=tuple(t.clone() for t in run["output"]),
                **{k: run["obs"][k].clone() for k in ("x", "ids", "tw", "moe")})


def _dv_correct(run, incumbent, ref, iso_ref, carry, bounds, cache_ref, data):
    obs, out = run["obs"], run["output"]
    ids_same = torch.equal(out[2], incumbent["output"][2])
    upstream = all(torch.equal(obs[k].view(torch.uint8), incumbent[k].view(torch.uint8)) for k in ("x", "ids", "tw"))
    cache_same = all(torch.equal(a, b) for a, b in zip(_vd_cache(data), cache_ref))
    errors, inc_errors = {}, {}
    for name, value, old, target in (
        ("moe", obs["moe"], incumbent["moe"], iso_ref),
        ("hidden", out[0], incumbent["output"][0], ref["hidden"]),
        ("update", out[1] - carry, incumbent["output"][1] - carry, ref["update"]),
    ):
        errors[name], inc_errors[name] = _dv_error(value, target), _dv_error(old, target)
    numeric = (ids_same and upstream and cache_same
               and all(_dv_numeric(errors[k], inc_errors[k], bounds[k]) for k in errors))
    return dict(numeric=bool(numeric), err_new=max(v[0] for v in errors.values()),
                err_inc=max(v[0] for v in inc_errors.values()), ids_equal=int(ids_same), upstream=int(upstream),
                cache_equal=int(cache_same), errors=errors, inc_errors=inc_errors)


def _dv_shape(st, weights, adapter, sliding, flush, sink, cache, deadline, reports):
    label, key = "d32" + ("S" if sliding else "F"), (32, 1, sliding)
    data, refresh = _vd_inputs(st, 32, 1, sliding, 20261104 + 7919 * int(sliding))
    if not st["ma_ok"] or not st["att_ok"] or not _supported7(data["config"]) or not _att_fits7(data, data["config"]):
        raise RuntimeError("native incumbent route unavailable")
    _dv_deadline(deadline)
    base, null, new = _dv_capture(data, st), _dv_capture(data, st), _dv_capture(data, st, adapter)
    if not new["obs"].get("candidate") or base["obs"].get("candidate") or null["obs"].get("candidate"):
        raise RuntimeError("MoE substitution was not observed in the full-layer graph")
    alpha1, _ = _dv_coeff(weights["attn_residual_gate"])
    alpha2, _ = _dv_coeff(weights["ffn_residual_gate"])
    rows = []
    for turn in range(DV_PASSES):
        _dv_deadline(deadline)
        refresh()
        before, snap = _vd_cache(data), _vd_snapshot(data)
        _dv_replay(base, data, snap, before)
        # Same busiest-range policy as the public input contract; no routing
        # modification, only choose this diagnostic input's local_start.
        busiest = torch.bincount(base["output"][2].long().flatten() // 32, minlength=8).argmax()
        data["local_start"].fill_(32 * int(busiest))
        snap = _vd_snapshot(data)
        carry = alpha1 * alpha2 * snap["stream"]
        _dv_replay(base, data, snap, before)
        inc = _dv_observations(base)
        cache_ref = _vd_cache(data)
        x, ids, tw = inc["x"], inc["ids"], inc["tw"]
        if (x.dtype != torch.bfloat16 or ids.dtype != torch.int32 or tw.dtype != torch.float32
                or x.shape != (32, 4096) or ids.shape != (32, 8)
                or not bool(((ids >= -1) & (ids < 32)).all())):
            raise RuntimeError("routing-to-VMS adapter contract mismatch")
        if not bool(torch.isfinite(x).all() & torch.isfinite(tw).all() & (tw >= 0).all()):
            raise RuntimeError("invalid routing input/weights")
        _vd_restore(data, snap, before)
        iso_ref = _dv_ref_moe(x, ids, tw, weights, cache, deadline)
        ref = _dv_reference(data, weights, inc["output"][2], cache, deadline)
        e4 = _dv_reference(data, weights, inc["output"][2], cache, deadline, fp8=True)
        bounds = {k: min(.15, max(.03, 2.5 * _dv_error(e4[k], ref[k])[0])) for k in ("hidden", "update")}
        bounds["moe"] = .015
        _dv_replay(new, data, snap, before)
        first = _dv_correct(new, inc, ref, iso_ref, carry, bounds, cache_ref, data)
        # Do not demand bit identity of the MoE/final output from either engine.
        timing = _dv_measure(base, null, new, data, snap, before, flush, sink, deadline)
        # Compare fresh replayed outputs of BOTH arms against the same references.
        _dv_replay(base, data, snap, before)
        last_inc = _dv_observations(base)
        _dv_replay(new, data, snap, before)
        last = _dv_correct(new, last_inc, ref, iso_ref, carry, bounds, cache_ref, data)
        last["numeric"] &= first["numeric"] and int(adapter.vs["err"].item()) == 0
        row = dict(timing, **last, shape=label, trial=turn + 1)
        rows.append(row)
        reports.append(row)
        print(_dv_line(row, False), flush=True)
    admitted = _dv_admit(rows)
    if admitted:
        st["dvms32_pick"][key] = adapter
    print(f"DEC_VMS32_PICK shape={label} arm=VMS32 admitted={int(admitted)} passes={len(rows)} "
          f"median_paired={_dv_stats.median(r['paired'] for r in rows):.6f}", flush=True)


def _dv_line(row, admitted):
    return (f"DEC_VMS32 shape={row['shape']} pass={row['trial']} old_us={row['old_us']:.3f} new_us={row['new_us']:.3f} "
            f"ratio={row['ratio']:.6f} paired={row['paired']:.6f} null={row['null']:.6f} null_ratio={row['null_ratio']:.6f} "
            f"err_new={row['err_new']:.6e} err_inc={row['err_inc']:.6e} admitted={int(admitted)} "
            f"numeric={int(row['numeric'])} ids_equal={row['ids_equal']} upstream={row['upstream']} "
            f"cache_equal={row['cache_equal']} errors={row['errors']} inc_errors={row['inc_errors']}")


def prepare(weights):
    global MFAB_LAYER
    start, old_mfab = _dv_time.monotonic(), MFAB_LAYER
    result, reports, cache = {}, [], {}
    thread = None
    def build():
        try:
            _vmsgate_static_check()
            result["ext"] = _vmsgate_ext(32)
        except Exception as ex:
            result["error"] = f"{type(ex).__name__}: {str(ex)[-900:]}"
    if torch.cuda.get_device_capability(weights["qkv_gate"].device) == (9, 0):
        thread = threading.Thread(target=build, daemon=True)
        thread.start()
    try:
        MFAB_LAYER = False  # incumbent's extra ablation is not part of its production route
        st = _DV_PREPARE(weights)
    finally:
        MFAB_LAYER = old_mfab
    st["dvms32_pick"] = {}
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        if thread is None or "-DOKC_PRMTSC=1" not in tuple(st.get("ma_sfx", ())):
            raise RuntimeError("requires SM90a and PRMT-exact incumbent scales")
        thread.join(timeout=max(0., start + DV_BUILD_END - _dv_time.monotonic()))
        if thread.is_alive() or "ext" not in result:
            raise RuntimeError(result.get("error", "optional VMS build deadline"))
        _dv_deadline(start + DV_GATE_END)
        ext = result["ext"]
        for kind in range(4):
            attrs = list(ext.vms_attr(kind, 0, 0))
            if not attrs[0]:
                raise RuntimeError(f"VMS32 kernel missing: kind={kind}")
            print(f"DEC_VMS32_BUILD kind={kind} regs_local_static_dyn={attrs[1:]}", flush=True)
        dev, w = st["qkv"].device, st["ma_w"]
        wA, wB = _vmsgate_repack(w[10], 64), _vmsgate_repack(w[11], 32)
        vs = dict(wA=wA, wB=wB, wA2=wA, wB2=wB, gsA=w[2], gsB=w[5], normw=w[7], lo=0, nloc=32, grid=int(st["nsm"]),
                  inter=torch.empty((256, 4096), dtype=torch.bfloat16, device=dev),
                  part=torch.empty((2, 256, 4096), dtype=torch.float32, device=dev),
                  ybuf=torch.empty((256, 4096), dtype=torch.float32, device=dev),
                  rtpos=torch.empty(256, dtype=torch.int32, device=dev), err=torch.zeros(1, dtype=torch.int32, device=dev))
        adapter = _DVAdapter(ext, vs)
        flush = torch.ones((512 << 20) // 4, dtype=torch.float32, device=dev)
        sink = torch.empty((), dtype=torch.float32, device=dev)
        for sliding in (True, False):
            label = "d32" + ("S" if sliding else "F")
            try:
                with torch.no_grad():
                    _dv_shape(st, weights, adapter, sliding, flush, sink, cache, start + DV_GATE_END, reports)
            except Exception as ex:
                st["dvms32_pick"].pop((32, 1, sliding), None)
                print(f"DEC_VMS32_PICK shape={label} arm=VMS32 admitted=0 "
                      f"error={type(ex).__name__}:{str(ex)[-900:]} fallback=incumbent", flush=True)
                torch.cuda.synchronize()
        if int(vs["err"].item()):
            st["dvms32_pick"].clear()
        _check_guards()
    except Exception as ex:
        st["dvms32_pick"].clear()
        print(f"DEC_VMS32_ABORT {type(ex).__name__}: {str(ex)[-1200:]} fallback=incumbent", flush=True)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
        cache.clear()
    # Tail-safe final table reports the actual shape pick, after all checks.
    for row in reports:
        admitted = (32, 1, row["shape"].endswith("S")) in st["dvms32_pick"]
        print(_dv_line(row, admitted), flush=True)
    for sliding in (True, False):
        print(f"DEC_VMS32_PICK shape=d32{'S' if sliding else 'F'} arm=VMS32 "
              f"admitted={int((32, 1, sliding) in st['dvms32_pick'])} prepare_s={_dv_time.monotonic()-start:.2f}", flush=True)
    return st
