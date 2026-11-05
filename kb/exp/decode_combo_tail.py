# DEC_COMBO: independent KV/POOL candidates, native incumbent by default.
import math as _kp_math
import statistics as _kp_stats
import time as _kp_time

_KP_PREPARE, _KP_KERNEL = prepare, kernel_fn
_KP_LAUNCH, _KP_ATTENTION = _launch7, _decode_attention7
_KP_TLS = threading.local()
_KP_KEYS = tuple((1, L, s) for L in (1, 8) for s in (True, False))
_DC_ROUNDS = 16

# INSERT_KVPRE_HELPERS


def _dc_arms(key):
    return ('KV',) if key[1] == 1 else ('POOL', 'KV+POOL')


def kernel_fn(data, state):
    arm = state.get('combo_pick', {}).get(_kp_key(data['config']), 'BASE')
    if not _kp_compatible(data):
        arm = 'BASE'
    saved = getattr(_KP_TLS, 'ext', None), getattr(_KP_TLS, 'observe', None)
    _KP_TLS.ext = state.get('kvpre_ext') if arm in ('KV', 'KV+POOL') else None
    _KP_TLS.observe = state.get('_kp_observe')
    try:
        return _KP_KERNEL(data, dict(state, _dp_active=arm in ('POOL', 'KV+POOL')))
    finally:
        _KP_TLS.ext, _KP_TLS.observe = saved


def _dc_layer(data, st, key, name):
    ss = _kp_private(st)
    ss['combo_pick'] = {} if name == 'BASE' else {key: name}
    ss['_kp_observe'] = {}
    def call():
        ss['_kp_observe'].clear()
        return kernel_fn(data, ss)
    graph, out = _kp_capture(call, st['qkv'].device)
    obs = ss['_kp_observe']
    if not {'qkv', 'attn', 'used', 'pool_used'} <= obs.keys():
        raise RuntimeError('native component not reached')
    if obs['used'] != (name in ('KV', 'KV+POOL')) or obs['pool_used'] != (name in ('POOL', 'KV+POOL')):
        raise RuntimeError('candidate component not reached')
    return dict(graph=graph, out=out, state=ss)


def _dc_snapshot(arm, data, carry):
    out = arm['out']
    vals = dict(hidden=out[0].clone(), stream=out[1].clone(), update=(out[1] - carry).clone())
    for name, t in zip(('k_new', 'v_new'), _kp_slots(data)):
        vals[name] = t.view(torch.float8_e4m3fn).view(1, data['config']['q_len'], 2, 64)
    return vals, out[2].clone()


def _dc_metrics(a, b):
    # checks.compare: global relative L2 and worst last-dimension row error.
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


def _dc_validate(refs, candidates, reference, tolerances, untouched):
    _, ids = refs[0]
    # Neither baseline variability nor a loose judge tolerance is permission to
    # change ordered routes. Any baseline ID instability fails this pass too.
    ids_equal = all(torch.equal(s[1], ids) for s in refs + candidates)
    envelope = {k: [0., 0.] for k in reference}
    within = bool(untouched)
    for vals, _ in refs:
        for k, want in reference.items():
            err = _dc_metrics(vals[k], want)
            within &= _dc_within(err, (tolerances[k], 5 * tolerances[k]))
            envelope[k] = [max(a, b) for a, b in zip(envelope[k], err)]
    for vals, _ in candidates:
        for k, want in reference.items():
            err = _dc_metrics(vals[k], want)
            within &= _dc_within(err, (tolerances[k], 5 * tolerances[k]))
            within &= _dc_within(err, envelope[k])
    return bool(ids_equal), bool(within)


def _dc_accept(records):
    if len(records) != 3:
        return False
    return bool(all(r['rounds'] >= 16 and r['abba'] >= 8 and r['baab'] >= 8
                    and r['ids_equal'] and r['within_tol']
                    and all(_kp_math.isfinite(r[k]) for k in ('ratio', 'paired', 'null', 'null_paired'))
                    and 0 < r['paired'] < 1.0
                    and 0.995 <= r['null'] <= 1.005 and 0.995 <= r['null_paired'] <= 1.005
                    for r in records) and _kp_stats.median(r['paired'] for r in records) <= .995)


def _dc_pick(records):
    admitted = [a for a, rs in records.items() if _dc_accept(rs)]
    return min(admitted, key=lambda a: (_kp_stats.median(r['paired'] for r in records[a]), a)) if admitted else 'BASE'


def _dc_budget(deadline, reserve=0):
    if _kp_time.monotonic() >= deadline - reserve:
        raise TimeoutError('DEC_COMBO incomplete trial discarded')


def _dc_pass(st, weights, key, passno, flush, sink, deadline):
    _dc_budget(deadline, 5)
    data = _kp_inputs(st, key, 20261104 + 1031 * passno + key[1] + 37 * int(key[2]))
    original = tuple(data[k].view(torch.uint8).clone() for k in ('k_cache', 'v_cache'))
    slots = _kp_slots(data)
    base = _dc_layer(data, st, key, 'BASE')
    null = _dc_layer(data, st, key, 'BASE')
    arms = {name: _dc_layer(data, st, key, name) for name in _dc_arms(key)}
    # Reference chooses the busiest local range just as the task does. Its math
    # never mutates caches and never participates in event timing or dispatch.
    front = _dc_ref_front(data, weights)
    ids = front['biased'].topk(8, dim=-1).indices
    busiest = int(torch.bincount((ids // 32).flatten(), minlength=8).argmax())
    data['local_start'].fill_(32 * busiest)
    front['local_start'] = 32 * busiest
    _kp_reset(data, slots)
    base['graph'].replay()
    torch.cuda.synchronize()
    ids = base['out'][2].clone()
    reference, tolerances, carry = _dc_reference(data, weights, front, ids, deadline)
    valid = _dc_ref_routing(ids, front['biased'], front['local_start'])
    # Include the exact incumbent replay whose IDs selected ref_back's experts.
    # Reference torch work does not overwrite this graph's output/cache rows.
    refs, samples = [_dc_snapshot(base, data, carry)], {name: [] for name in arms}
    valid &= _kp_untouched(data, original)
    events = tuple(torch.cuda.Event(enable_timing=True) for _ in range(2))
    blocks, nulls = {name: [] for name in arms}, []

    def sample(arm, target):
        nonlocal valid
        _kp_reset(data, slots)
        torch.sum(flush, dim=0, out=sink)
        arm['graph'].replay()
        torch.cuda.synchronize()
        target.append(_dc_snapshot(arm, data, carry))
        valid &= _kp_untouched(data, original)

    for arm in (base, null, null, base):
        sample(arm, refs)
    # All arms share a contemporaneous null but have their own ABBA/BAAB
    # comparisons against BASE. Rotate order between rounds and fresh passes.
    for r in range(_DC_ROUNDS):
        _dc_budget(deadline, 2)
        order = (0, 1, 1, 0) if (r + passno) % 2 == 0 else (1, 0, 0, 1)
        names = ['NULL'] + list(arms)
        shift = (r + passno) % len(names)
        names = names[shift:] + names[:shift]
        if (r + passno) % 2:
            names.reverse()
        for name in names:
            arm = null if name == 'NULL' else arms[name]
            target = nulls if name == 'NULL' else blocks[name]
            target.append(_kp_block(base['graph'], arm['graph'], order, data, slots, flush, sink, events))
        # 21 incumbent samples and four samples per arm. Whole-cache checking
        # is outside event intervals; independent input seeds in each pass.
        sample(base if r % 2 == 0 else null, refs)
        if r % 4 == 3:
            for name, arm in arms.items():
                sample(arm, samples[name])
    _dc_budget(deadline)
    _, _, nr, np = _kp_summarize(nulls)
    result = {}
    for name in arms:
        ids_equal, within = _dc_validate(refs, samples[name], reference, tolerances, valid)
        old, new, ratio, paired = _kp_summarize(blocks[name])
        result[name] = dict(old_us=old, new_us=new, ratio=ratio, paired=paired, null=nr, null_paired=np,
                            ids_equal=ids_equal, within_tol=within, rounds=len(blocks[name]),
                            abba=sum((r + passno) % 2 == 0 for r in range(_DC_ROUNDS)),
                            baab=sum((r + passno) % 2 != 0 for r in range(_DC_ROUNDS)))
    _check_guards()
    _dc_budget(deadline)
    return result


def _dc_gate(st, weights, deadline):
    global _WS7, _CL7, _GUARDS, _GUARDED_IDS, bfp_SK2_PDL, _PF7, _GLE_W_CUR
    saved = _WS7, _CL7, _GUARDS, _GUARDED_IDS, bfp_SK2_PDL, _PF7, _GLE_W_CUR
    _WS7, _CL7, _GUARDS, _GUARDED_IDS = {}, {}, [], set()
    picks = {}
    try:
        flush = torch.ones((512 << 20) // 4, dtype=torch.float32, device=st['qkv'].device)
        sink = torch.empty((), dtype=torch.float32, device=flush.device)
        for key in _KP_KEYS:
            label = f'{"d" if key[1] == 1 else "v"}1{"S" if key[2] else "F"}'
            records = {name: [] for name in _dc_arms(key)}
            try:
                for p in range(3):
                    for name, r in _dc_pass(st, weights, key, p, flush, sink, deadline).items():
                        records[name].append(r)
                pick = _dc_pick(records)
                if pick != 'BASE':
                    picks[key] = pick
            except Exception as ex:
                print(f'DEC_COMBO_ERROR shape={label} {type(ex).__name__}:{str(ex)[-250:]}', flush=True)
                torch.cuda.synchronize()
            for name, rs in records.items():
                for p, r in enumerate(rs):
                    print(f'DEC_COMBO shape={label} arm={name} pass={p+1} old_us={r["old_us"]:.3f} '
                          f'new_us={r["new_us"]:.3f} ratio={r["ratio"]:.6f} paired={r["paired"]:.6f} '
                          f'null={r["null"]:.6f} null_paired={r["null_paired"]:.6f} '
                          f'ids_equal={int(r["ids_equal"])} within_tol={int(r["within_tol"])} '
                          f'admitted={int(_dc_accept(rs))} rounds={r["rounds"]}', flush=True)
                if len(rs) < 3:
                    print(f'DEC_COMBO shape={label} arm={name} pass={len(rs)+1} old_us=nan new_us=nan '
                          'ratio=nan paired=nan null=nan ids_equal=0 within_tol=0 admitted=0 incomplete=1', flush=True)
            print(f'DEC_COMBO_PICK shape={label} arm={picks.get(key, "BASE")}', flush=True)
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
    st['combo_pick'] = {}
    try:
        for thread in (kv_thread, pool_thread):
            thread.join(timeout=max(0., start + 230. - _kp_time.monotonic()))
        if any(t.is_alive() for t in (kv_thread, pool_thread)) or 'ext' not in mailbox or _DP_EXT is None:
            raise RuntimeError('build unavailable ' + mailbox.get('error', '') + ' '.join(_DP_ERROR))
        _dc_budget(start + 275., 10)
        st['kvpre_ext'] = _RouterExt(mailbox['ext'])
        st['_dp_weight'] = _dp_pack(st['router_fp32'])
        if not torch.equal(_dp_unpack(st['_dp_weight']).view(torch.uint8), st['router_fp32'].view(torch.uint8)):
            raise RuntimeError('router pack roundtrip mismatch')
        if not st.get('att_ok') or st.get('bfp') is None:
            raise RuntimeError('native incumbent unavailable')
        with torch.no_grad():
            picks = _dc_gate(st, weights, start + 275.)
            _check_guards()
        st['combo_pick'] = picks
    except Exception as ex:
        st['combo_pick'] = {}
        print(f'DEC_COMBO_ERROR gate_aborted {type(ex).__name__}:{str(ex)[-400:]} fallback=incumbent', flush=True)
        torch.cuda.synchronize()
        for key in _KP_KEYS:
            label = f'{"d" if key[1] == 1 else "v"}1{"S" if key[2] else "F"}'
            print(f'DEC_COMBO_PICK shape={label} arm=BASE aborted=1', flush=True)
    print(f'DEC_COMBO_DONE prepare_s={_kp_time.monotonic()-start:.2f} admitted={len(st["combo_pick"])} '
          'passes=3 rounds=16 flush=READ512', flush=True)
    return st
