
# ---------------------------------------------------------------------------
# Idle-DRAM L2 weight prefetch (lane omos/decode-fresh-20261029).
#
# While the attention kernel and the attention boundary are latency-bound,
# HBM is mostly idle. A one-launch side-stream kernel issues
# prefetch.global.L2 for the exact weight buffers that the next projections
# read (output projection, then router, then a leading part of the pooler).
# Prefetch is a cache hint only: every arithmetic operation, launch order,
# dtype boundary and output of the retained path is unchanged. Buffers are
# discovered at prepare time by recording the leaf launch helpers that the
# retained dispatch actually calls for each token count.
# ---------------------------------------------------------------------------
import triton as _pf_triton
import triton.language as _pf_tl

_PF_CHUNK = 4096          # bytes per warp iteration: 32 lanes x 32 B x 4 prefetches
_PF_WARPS = 2
_PF_PROGRAMS = 32
_PF_BUDGET = 40 << 20     # bytes per call at most (H200 L2 is 50 MB)
_PF_MAX_TOKENS = 256
_PF_MIN_BYTES = 1 << 20
_PF_RECORD = None


@_pf_triton.jit
def _pf_l2_kernel(Tab, WARPS: _pf_tl.constexpr):
    pid = _pf_tl.program_id(0).to(_pf_tl.int64)
    nprog = _pf_tl.num_programs(0).to(_pf_tl.int64)
    e = _pf_tl.arange(0, WARPS * 32)
    lane = (e % 32).to(_pf_tl.int64)
    wid = (e // 32).to(_pf_tl.int64)
    b0 = _pf_tl.load(Tab + 0)
    b1 = _pf_tl.load(Tab + 1)
    b2 = _pf_tl.load(Tab + 2)
    b3 = _pf_tl.load(Tab + 3)
    c1 = _pf_tl.load(Tab + 4)
    c2 = _pf_tl.load(Tab + 5)
    c3 = _pf_tl.load(Tab + 6)
    n = _pf_tl.load(Tab + 7)
    step = nprog * WARPS
    iters = (n + step - 1) // step
    # Chunk c goes to warp c % step in iteration c // step, so the whole grid
    # walks the buffers in priority order (output projection first).
    for it in range(0, iters):
        c = _pf_tl.minimum(it * step + pid * WARPS + wid, n - 1)
        a = _pf_tl.where(c < c1, b0 + c * 4096,
                         _pf_tl.where(c < c2, b1 + (c - c1) * 4096,
                                      _pf_tl.where(c < c3, b2 + (c - c2) * 4096, b3 + (c - c3) * 4096)))
        _pf_tl.inline_asm_elementwise(
            "prefetch.global.L2 [$1];\n\tprefetch.global.L2 [$1+1024];\n\t"
            "prefetch.global.L2 [$1+2048];\n\tprefetch.global.L2 [$1+3072];\n\tmov.u32 $0, 0;",
            "=r,l", [a + lane * 32], dtype=_pf_tl.int32, is_pure=False, pack=1)


def _pf_note(t):
    if _PF_RECORD is None or not isinstance(t, torch.Tensor) or not t.is_cuda or not t.is_contiguous():
        return
    nbytes = t.numel() * t.element_size()
    if nbytes >= _PF_MIN_BYTES:
        _PF_RECORD.append((int(t.data_ptr()), int(nbytes)))


def _pf_wrap_dict_args(fn):
    def wrapped(*args, **kwargs):
        if _PF_RECORD is not None:
            for a in tuple(args) + tuple(kwargs.values()):
                if isinstance(a, dict):
                    _pf_note(a.get('wp'))
                    _pf_note(a.get('w'))
        return fn(*args, **kwargs)
    return wrapped


def _pf_wrap_olayer(fn):
    def wrapped(x, st, index):
        if _PF_RECORD is not None:
            pack = st['configs'][index][4]
            _pf_note(st.get('tiled32') if pack == 2 else st.get('packed') if pack == 1 else st.get('weight'))
        return fn(x, st, index)
    return wrapped


def _pf_wrap_second(fn):
    def wrapped(x, w, *args, **kwargs):
        _pf_note(w)
        return fn(x, w, *args, **kwargs)
    return wrapped


def _pf_wrap_routerwide(fn):
    def wrapped(x, st, arm):
        if _PF_RECORD is not None:
            _pf_note(st.get('packed'))
        return fn(x, st, arm)
    return wrapped


# Leaf launch helpers of the retained dense dispatch: each receives the single
# selected plan (sp/pk/plan) or weight. Pass-through wrappers record only while
# _PF_RECORD is set during prepare-time discovery.
sm_launch_raw = _pf_wrap_dict_args(sm_launch_raw)
m32_launch_raw = _pf_wrap_dict_args(m32_launch_raw)
qkv_sm_launch_raw = _pf_wrap_dict_args(qkv_sm_launch_raw)
cps32_launch_raw = _pf_wrap_dict_args(cps32_launch_raw)
cpssm_launch_raw = _pf_wrap_dict_args(cpssm_launch_raw)
cps_launch = _pf_wrap_dict_args(cps_launch)
tt_mw_launch_raw = _pf_wrap_dict_args(tt_mw_launch_raw)
tt_v10_sm2_smallm2_launch_raw = _pf_wrap_dict_args(tt_v10_sm2_smallm2_launch_raw)
tt_v10_warps_warps_launch_raw = _pf_wrap_dict_args(tt_v10_warps_warps_launch_raw)
tt_v10_m32x_m32x_launch_raw = _pf_wrap_dict_args(tt_v10_m32x_m32x_launch_raw)
tt_v10_m32x__w_launch_raw = _pf_wrap_dict_args(tt_v10_m32x__w_launch_raw)
tt_v10_m32x__q_launch_raw = _pf_wrap_dict_args(tt_v10_m32x__q_launch_raw)
tt_v12_smtrace1_smtrace_launch_raw = _pf_wrap_dict_args(tt_v12_smtrace1_smtrace_launch_raw)
tt_v13_smtrace8_smtrace_launch_raw = _pf_wrap_dict_args(tt_v13_smtrace8_smtrace_launch_raw)
dd_decodediet_dd_cap2_b_cpssm_launch_raw = _pf_wrap_dict_args(dd_decodediet_dd_cap2_b_cpssm_launch_raw)
dd_decodediet_cps_launch = _pf_wrap_dict_args(dd_decodediet_cps_launch)
dd_decodediet_tt_mw_launch_raw = _pf_wrap_dict_args(dd_decodediet_tt_mw_launch_raw)
mt__launch = _pf_wrap_dict_args(mt__launch)
b2p_tt_v10_m32x_m32x_launch_raw = _pf_wrap_dict_args(b2p_tt_v10_m32x_m32x_launch_raw)
b2p_tt_v10_m32x__w_launch_raw = _pf_wrap_dict_args(b2p_tt_v10_m32x__w_launch_raw)
b2p_tt_v10_m32x__q_launch_raw = _pf_wrap_dict_args(b2p_tt_v10_m32x__q_launch_raw)
b2q_tt_v10_m32x__q_launch_raw = _pf_wrap_dict_args(b2q_tt_v10_m32x__q_launch_raw)
v18d_v18d_native_launch_raw = _pf_wrap_dict_args(v18d_v18d_native_launch_raw)
v18s_launch_raw = _pf_wrap_dict_args(v18s_launch_raw)
run_olayer_candidate = _pf_wrap_olayer(run_olayer_candidate)
b2o_run_olayer_candidate = _pf_wrap_olayer(b2o_run_olayer_candidate)
tt_v10_tune_run_candidate = _pf_wrap_second(tt_v10_tune_run_candidate)
rw_run_routerwide_candidate = _pf_wrap_routerwide(rw_run_routerwide_candidate)
_gemm = _pf_wrap_second(_gemm)


def _pf_record(call, *args):
    """Run one retained dispatch call and return the (address, bytes) spans of the
    weight buffers its leaf launch helpers pass to the device, in call order."""
    global _PF_RECORD
    _PF_RECORD = []
    try:
        call(*args)
        torch.cuda.synchronize()
    finally:
        (found, _PF_RECORD) = (_PF_RECORD, None)
    spans = []
    for span in found:
        if span not in spans:
            spans.append(span)
    return spans


def _pf_span(t):
    if isinstance(t, torch.Tensor) and t.is_cuda and t.is_contiguous():
        return (int(t.data_ptr()), int(t.numel() * t.element_size()))
    return None


def _pf_plan(state, m, dev):
    x = torch.zeros((m, 1024), dtype=torch.bfloat16, device=dev)
    spans = _pf_record(_achita_dense_project, x, 'o', state)
    pool = []
    if 8 < m <= 32 and state.get('fpr') is not None:
        sk2 = state['fpr']['sk2']
        router = [_pf_span(sk2['router'][1])]
        pool = [_pf_span(sk2['pooler'][1])]
    elif m <= ROUTER_FP32_MAX_TOKENS:
        rf = state.get('rf')
        router = [_pf_span(rf['weight'] if rf is not None else state['router'])]
    else:
        xr = torch.zeros((m, HIDDEN), dtype=torch.bfloat16, device=dev)
        router = _pf_record(_achita_wide_router, xr, state)
        if not router:
            router = [_pf_span(state['router_hi_t']), _pf_span(state['router_lo_t'])]
    if not pool:
        xp = torch.zeros((m, HIDDEN), dtype=torch.bfloat16, device=dev)
        pool = _pf_record(_achita_dense_project, xp, 'pooler', state)
    (segments, used, left) = ([], set(), _PF_BUDGET)
    for span in spans + router + pool:
        if span is None or span[0] in used or span[0] % 16 or len(segments) == 4:
            continue
        take = min(span[1], left) // _PF_CHUNK
        if take <= 0:
            continue
        segments.append((span[0], take))
        used.add(span[0])
        left -= take * _PF_CHUNK
    if not segments:
        return None
    (bases, cum, total) = ([], [], 0)
    for (base, count) in segments:
        bases.append(base)
        total += count
        cum.append(total)
    bases += [bases[0]] * (4 - len(bases))
    cum += [total] * (4 - len(cum))
    tab = torch.tensor(bases + cum[:3] + [total], dtype=torch.int64, device=dev)
    grid = max(1, min(_PF_PROGRAMS, (total + _PF_WARPS - 1) // _PF_WARPS))
    return {'tab': tab, 'grid': grid, 'bytes': total * _PF_CHUNK,
            'segments': [(count * _PF_CHUNK) for (_, count) in segments]}


def _pf_launch(plan):
    _pf_l2_kernel[(plan['grid'],)](plan['tab'], WARPS=_PF_WARPS, num_warps=_PF_WARPS)


def _pf_probe(state, plans, dev):
    """Cold-L2 diagnostic printed to the run log: output projection and pooler,
    each without and with a completed prefetch of its own plan."""
    flush = torch.empty(256 << 20, dtype=torch.uint8, device=dev)
    report = {}
    for m in (1, 8):
        plan = plans.get(m)
        if plan is None:
            continue
        xo = torch.randn((m, 1024), device=dev).to(torch.bfloat16)
        xp = torch.randn((m, HIDDEN), device=dev).to(torch.bfloat16)
        for (name, run) in (('o', lambda: _achita_dense_project(xo, 'o', state)),
                            ('pooler', lambda: _achita_dense_project(xp, 'pooler', state))):
            for warm in (False, True):
                samples = []
                for _ in range(7):
                    flush.sum()
                    if warm:
                        _pf_launch(plan)
                    torch.cuda.synchronize()
                    begin = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    begin.record()
                    run()
                    end.record()
                    end.synchronize()
                    samples.append(begin.elapsed_time(end) * 1000.0)
                report[f'm{m}_{name}_{"pf" if warm else "cold"}_us'] = round(sorted(samples)[3], 2)
    del flush
    torch.cuda.synchronize()
    return report


def _pf_prepare(state):
    if torch.cuda.get_device_capability()[0] != 9:
        return None
    import time as _pf_time
    started = _pf_time.monotonic()
    dev = state['router'].device
    plans = {}
    for m in range(1, _PF_MAX_TOKENS + 1):
        try:
            plan = _pf_plan(state, m, dev)
        except Exception as exc:
            print(f'[l2pf] m={m} skipped: {type(exc).__name__}: {exc}', flush=True)
            plan = None
        if plan is not None:
            plans[m] = plan
    if not plans:
        return None
    _pf_launch(plans[min(plans)])
    torch.cuda.synchronize()
    summary = {m: [round(b / 2 ** 20, 2) for b in plans[m]['segments']] for m in (1, 8, 16, 32, 128, 256) if m in plans}
    print(f'[l2pf] plans={len(plans)} segments_MiB={summary} prepare_s={_pf_time.monotonic() - started:.2f}', flush=True)
    try:
        print(f'[l2pf] probe={_pf_probe(state, plans, dev)}', flush=True)
    except Exception as exc:
        print(f'[l2pf] probe skipped: {type(exc).__name__}: {exc}', flush=True)
    return {'plans': plans, 'stream': torch.cuda.Stream(dev)}


def _pf_begin(state, cfg, tokens):
    pf = state.get('pf')
    if pf is None or not (cfg['window'] or cfg['batch'] == 1):
        return None
    plan = pf['plans'].get(tokens)
    if plan is None:
        return None
    (main, side) = (torch.cuda.current_stream(), pf['stream'])
    side.wait_stream(main)
    with torch.cuda.stream(side):
        _pf_launch(plan)
    return (main, side)


def kernel_fn(data, state):
    cfg = data['config']
    (batch, length, tokens) = (cfg['batch'], cfg['q_len'], cfg['tokens'])
    (hq, hkv, d) = (cfg['num_q_heads'], cfg['num_kv_heads'], cfg['head_dim'])
    qkv = _achita_dense_project(data['attn_input'], 'qkv_gate', state)
    prefetch = _pf_begin(state, cfg, tokens)
    attention_data = dict(data, qkv=qkv.reshape(batch, length, -1))
    o = _native_attn_kernel_fn(attention_data).reshape(tokens, hq * d)
    attn_output = _achita_dense_project(o, 'o', state)
    split_router = tokens > ROUTER_FP32_MAX_TOKENS
    (ffn_input, stream, router_input) = state['attn_boundary'](data['stream'], attn_output, branch_dtype=torch.bfloat16, router_dtype=None if split_router else torch.float32)
    (main_stream, side_stream) = (torch.cuda.current_stream(), state['side_stream'])
    if 8 < tokens <= 32 and state['fpr'] is not None:
        (expert_input, logits) = _fpr_bfp_fused_pooler_router(ffn_input, state['fpr'])
        (topk_weights, topk_ids) = state['routing'](None, logits, TOP_K, True)
    else:
        side_stream.wait_stream(main_stream)
        expert_input = _achita_dense_project(ffn_input, 'pooler', state)
        with torch.cuda.stream(side_stream):
            if not split_router and tokens > 0 and (state['rf'] is not None):
                (topk_weights, topk_ids) = _rf_route(router_input, state['rf'])
            else:
                if split_router:
                    logits = _achita_wide_router(ffn_input, state)
                    if logits is None:
                        logits = torch.mm(ffn_input, state['router_hi_t'], out_dtype=torch.float32)
                        logits += torch.mm(ffn_input, state['router_lo_t'], out_dtype=torch.float32)
                else:
                    logits = F.linear(router_input, state['router'])
                (topk_weights, topk_ids) = state['routing'](None, logits, TOP_K, True)
        main_stream.wait_stream(side_stream)
        if not torch.cuda.is_current_stream_capturing():
            for tensor in (topk_weights, topk_ids):
                tensor.record_stream(main_stream)
    expert_input, local_ids = _pn_expert_inputs(expert_input, topk_ids, data['local_start'], state)
    expert_output = _experts(expert_input, topk_weights, local_ids, state)
    if tokens <= SLICED_UNPOOL_MAX_TOKENS:
        unpooled = _achita_dense_project(expert_output, 'unpooler', state, UNPOOL_SLICE)
        next_attn_input, stream = _sfn_boundary(stream, unpooled, state)
    else:
        ffn_output = _achita_dense_project(expert_output, 'unpooler', state)
        next_attn_input, stream, _ = state['ffn_boundary'](stream, ffn_output, branch_dtype=torch.bfloat16)
    if prefetch is not None:
        prefetch[0].wait_stream(prefetch[1])
    return (next_attn_input, stream, topk_ids)


_pf_original_prepare = prepare


def prepare(weights):
    state = _pf_original_prepare(weights)
    state['pf'] = None
    try:
        state['pf'] = _pf_prepare(state)
    except Exception as exc:
        print(f'[l2pf] disabled: {type(exc).__name__}: {exc}', flush=True)
        state['pf'] = None
    return state
