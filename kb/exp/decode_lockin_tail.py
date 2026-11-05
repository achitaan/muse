def kernel_fn(data, state):
    key = _kp_key(data['config'])
    arm = state.get('lockin_pick', {}).get(key, 'BASE') if key in _DL_DEFAULTS else 'BASE'
    if not _kp_compatible(data):
        arm = 'BASE'
    saved = getattr(_KP_TLS, 'ext', None), getattr(_KP_TLS, 'observe', None)
    _KP_TLS.ext = state.get('kvpre_ext') if arm == 'KV' else None
    _KP_TLS.observe = state.get('_kp_observe')
    try:
        return _KP_KERNEL(data, dict(state, _dp_active=arm == 'POOL'))
    finally:
        _KP_TLS.ext, _KP_TLS.observe = saved


def _dl_layer(data, st, key, name):
    # Exactly one synthetic invocation per implementation, no graph warmups,
    # timing events, flush buffers, nulls, or speed-dependent selection.
    ss = _kp_private(st)
    ss['lockin_pick'] = {} if name == 'BASE' else {key: name}
    ss['_kp_observe'] = {}
    out = kernel_fn(data, ss)
    torch.cuda.synchronize()
    obs = ss['_kp_observe']
    if not {'qkv', 'attn', 'used', 'pool_used'} <= obs.keys():
        raise RuntimeError('native component not reached')
    if obs['used'] != (name == 'KV') or obs['pool_used'] != (name == 'POOL'):
        raise RuntimeError('candidate component not reached')
    return out


def _dl_snapshot(out, data, carry):
    vals = dict(hidden=out[0].clone(), stream=out[1].clone(), update=(out[1] - carry).clone())
    for name, t in zip(('k_new', 'v_new'), _kp_slots(data)):
        vals[name] = t.view(torch.float8_e4m3fn).view(1, data['config']['q_len'], 2, 64)
    return vals, out[2].clone()


def _dc_metrics(a, b):
    # Public checks.compare: relative L2 and worst last-dimension row error.
    if a.shape != b.shape or a.dtype != b.dtype:
        return float('inf'), float('inf')
    a, b = a.float(), b.float()
    if not bool(torch.isfinite(a).all() & torch.isfinite(b).all()):
        return float('inf'), float('inf')
    diff = a - b
    whole = diff.norm() / b.norm().clamp_min(1e-12)
    rn = b.reshape(-1, b.shape[-1]).norm(dim=-1)
    dn = diff.reshape(-1, b.shape[-1]).norm(dim=-1)
    rows = dn / torch.maximum(rn, (0.1 * rn.mean()).clamp_min(1e-12))
    return float(whole), float(rows.max())


def _dc_within(errors, bounds):
    return all(_kp_math.isfinite(e) and e <= b for e, b in zip(errors, bounds))


def _dl_validate(inc, new, reference, tolerances, untouched, routing):
    ids_equal = bool(torch.equal(inc[1], new[1]))
    err_inc = {k: _dc_metrics(inc[0][k], v) for k, v in reference.items()}
    err_new = {k: _dc_metrics(new[0][k], v) for k, v in reference.items()}
    reasons = []
    if not ids_equal:
        reasons.append('expert_ids')
    if not untouched:
        reasons.append('untouched_cache')
    if not routing:
        reasons.append('incumbent_routing')
    for k in reference:
        bound = (tolerances[k], 5 * tolerances[k])
        if not _dc_within(err_inc[k], bound):
            reasons.append('inc_judge_' + k)
        if not _dc_within(err_new[k], bound):
            reasons.append('new_judge_' + k)
        # No epsilon/floor or repeat envelope: compare against this one call.
        if not _dc_within(err_new[k], err_inc[k]):
            reasons.append('worse_' + k)
    return ids_equal, err_new, err_inc, not reasons, ','.join(reasons) or 'ok'


def _dc_budget(deadline, reserve=0):
    if _kp_time.monotonic() >= deadline - reserve:
        raise TimeoutError('DEC_LOCKIN correctness check deadline')


def _dl_check(st, weights, key, arm, deadline):
    _dc_budget(deadline, 5)
    data = _kp_inputs(st, key, 20261104 + key[1] + 37 * int(key[2]))
    original = tuple(data[k].view(torch.uint8).clone() for k in ('k_cache', 'v_cache'))
    slots = _kp_slots(data)
    front = _dc_ref_front(data, weights)
    ids = front['biased'].topk(8, dim=-1).indices
    busiest = int(torch.bincount((ids // 32).flatten(), minlength=8).argmax())
    data['local_start'].fill_(32 * busiest)
    front['local_start'] = 32 * busiest
    a1, _ = _dc_ref_coeff(weights['attn_residual_gate'])
    a2, _ = _dc_ref_coeff(weights['ffn_residual_gate'])
    carry = a1 * a2 * data['stream'].float()
    _kp_reset(data, slots)
    inc = _dl_snapshot(_dl_layer(data, st, key, 'BASE'), data, carry)
    untouched = _kp_untouched(data, original)
    routing = _dc_ref_routing(inc[1], front['biased'], front['local_start'])
    reference, tolerances, _ = _dc_reference(data, weights, front, inc[1], deadline)
    _dc_budget(deadline, 2)
    _kp_reset(data, slots)
    new = _dl_snapshot(_dl_layer(data, st, key, arm), data, carry)
    untouched &= _kp_untouched(data, original)
    result = _dl_validate(inc, new, reference, tolerances, untouched, routing)
    _check_guards()
    _dc_budget(deadline)
    return result


def _dl_safety(st, weights, deadline):
    global _WS7, _CL7, _GUARDS, _GUARDED_IDS, bfp_SK2_PDL, _PF7, _GLE_W_CUR
    saved = _WS7, _CL7, _GUARDS, _GUARDED_IDS, bfp_SK2_PDL, _PF7, _GLE_W_CUR
    _WS7, _CL7, _GUARDS, _GUARDED_IDS = {}, {}, [], set()
    picks = {}
    try:
        for key, arm in _DL_DEFAULTS.items():
            label = 'd1F' if key[1] == 1 else 'v1S'
            ids_equal, err_new, err_inc, used, why = False, {}, {}, False, 'incomplete'
            try:
                if arm in st.get('_dl_unavailable', {}):
                    raise RuntimeError(st['_dl_unavailable'][arm])
                ids_equal, err_new, err_inc, used, why = _dl_check(st, weights, key, arm, deadline)
                if used:
                    picks[key] = arm
            except Exception as ex:
                why = f'{type(ex).__name__}:{str(ex)[-350:]}'
                torch.cuda.synchronize()
            print(f'DEC_LOCKIN shape={label} arm={arm} ids_equal={int(ids_equal)} '
                  f'err_new={err_new} err_inc={err_inc} used={arm if used else "BASE"} why={why}', flush=True)
        _check_guards()
    finally:
        try:
            torch.cuda.synchronize()
        finally:
            _WS7, _CL7, _GUARDS, _GUARDED_IDS, bfp_SK2_PDL, _PF7, _GLE_W_CUR = saved
            bfp__get_ext().sk2_set_pdl(bfp_SK2_PDL)
            for ext in (_GLE_EXT, _DE_EXT, _DP_EXT):
                if ext is not None:
                    ext.sk2_set_pdl(bfp_SK2_PDL)
            for ext in st['ma_ext'].values():
                ext.okc_set_pdl(0)
    return picks


def _dc_pool_build(device):
    torch.cuda.set_device(device)
    _dp_build()


def prepare(weights):
    start = _kp_time.monotonic()
    device = weights['qkv_gate'].device
    mailbox, kv_thread = _kp_start_build(device)
    pool_thread = threading.Thread(target=_dc_pool_build, args=(device,), daemon=True)
    pool_thread.start()
    st = _KP_PREPARE(weights)
    st['lockin_pick'] = {}
    st['_dl_unavailable'] = {}
    try:
        for thread in (kv_thread, pool_thread):
            thread.join(timeout=max(0., start + 230. - _kp_time.monotonic()))
        if kv_thread.is_alive() or 'ext' not in mailbox:
            st['_dl_unavailable']['KV'] = 'build unavailable ' + mailbox.get('error', 'timeout')
        else:
            st['kvpre_ext'] = _RouterExt(mailbox['ext'])
        if pool_thread.is_alive() or _DP_EXT is None:
            st['_dl_unavailable']['POOL'] = 'build unavailable ' + ' '.join(_DP_ERROR)
        else:
            st['_dp_weight'] = _dp_pack(st['router_fp32'])
            if not torch.equal(_dp_unpack(st['_dp_weight']).view(torch.uint8), st['router_fp32'].view(torch.uint8)):
                st['_dl_unavailable']['POOL'] = 'router pack roundtrip mismatch'
        if not st.get('att_ok') or st.get('bfp') is None:
            raise RuntimeError('native incumbent unavailable')
        with torch.no_grad():
            picks = _dl_safety(st, weights, start + 275.)
            _check_guards()
        st['lockin_pick'] = picks
    except Exception as ex:
        st['lockin_pick'] = {}
        print(f'DEC_LOCKIN_ERROR {type(ex).__name__}:{str(ex)[-400:]} fallback=incumbent', flush=True)
        torch.cuda.synchronize()
        for key, arm in _DL_DEFAULTS.items():
            label = 'd1F' if key[1] == 1 else 'v1S'
            print(f'DEC_LOCKIN shape={label} arm={arm} ids_equal=0 err_new={{}} err_inc={{}} '
                  'used=BASE why=prepare_aborted', flush=True)
    print(f'DEC_LOCKIN_DONE prepare_s={_kp_time.monotonic()-start:.2f} '
          f'admitted={len(st["lockin_pick"])} checks=one_per_shape no_timing=1', flush=True)
    return st
