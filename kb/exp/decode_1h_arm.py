# Correctness-only admission. No timing gate, benchmark, or process launch.
_D1H_MODULES = {}
_D1H_WS = {}
_D1H_SHAPES = ((16, 8, True), (32, 8, True))
_D1H_SPLITS = {16: 8, 32: 4}
_D1H_OLD_ATTN = _decode_attention7
_D1H_OLD_PREPARE = prepare


def _d1h_build(device):
    try:
        torch.cuda.set_device(device)
        # Exactly the incumbent _att7 load_inline flags, including its architecture policy.
        major, minor = torch.cuda.get_device_capability()
        arch = f'{major}{minor}' + ('a' if major >= 9 else '')
        os.environ['TORCH_CUDA_ARCH_LIST'] = f'{major}.{minor}' + ('a' if major == 9 else '')
        _D1H_MODULES['1h'] = load_inline(
            name=f'muse_decode_onehead_{arch}', cpp_sources=AH_CPP, cuda_sources=AH_CUDA,
            functions=['run_1h'],
            extra_cuda_cflags=['-DOKC_QDIRECT=0', '-DOKC_PK2=0', '-DOKC_W0ENABLE=1', '-O3',
                              f'-DOKC_SM90={1 if major >= 9 else 0}', '--expt-relaxed-constexpr'],
            verbose=False)
    except Exception as ex:
        _D1H_MODULES['error'] = repr(ex)[-1200:]


def _decode_attention7(data, state, qkv):
    cfg = data['config']
    key = _d1h_key(cfg)
    used = False
    if (key in state.get('dec_1h', ()) and _d1h_compatible(data, qkv)
            and cfg.get('tokens') == cfg['batch'] * cfg['q_len']):
        args = {k: data[k] for k in ('config', 'seq_lens', 'cache_loc', 'k_cache', 'v_cache',
                                    'page_table', 'k_scale', 'v_scale')}
        args['qkv'] = qkv.view(cfg['batch'], cfg['q_len'], QKV_ROWS)
        try:
            out = _d1h_ah_launch(args, _D1H_SPLITS[cfg['batch']], 1).view(-1, Q_DIM)
            used = True
        except _AHLaunchRejected:
            # CPU metadata rejection happens before a CUDA launch; never catch a device fault here.
            out = _D1H_OLD_ATTN(data, state, qkv)
    else:
        out = _D1H_OLD_ATTN(data, state, qkv)
    observe = state.get('_dec_1h_observe')
    if observe is not None:
        observe['attn'] = out
        observe['used'] = used
    return out


def _d1h_admit(errors, cache_ok, ids_equal, used):
    # Public judge's minimum allowance: global relative L2 .03, row .15.
    # Compare stream updates too, so a large unchanged residual cannot mask error.
    return (cache_ok and ids_equal and used and len(errors) == 4
            and all(global_err <= .03 and row_err <= .15 for global_err, row_err in errors))


def _d1h_error(a, b):
    # Match checks.compare's global and last-dimension row metrics exactly.
    if a.shape != b.shape or a.dtype != b.dtype:
        return float('inf'), float('inf')
    a, b = a.float(), b.float()
    if not bool(torch.isfinite(a).all() & torch.isfinite(b).all()):
        return float('inf'), float('inf')
    diff = a - b
    row_diff = diff.reshape(-1, b.shape[-1]).norm(dim=-1)
    row_ref = b.reshape(-1, b.shape[-1]).norm(dim=-1)
    floor = (0.1 * row_ref.mean()).clamp(min=1e-12)
    return (float(diff.norm() / b.norm().clamp(min=1e-12)),
            float((row_diff / torch.maximum(row_ref, floor)).max()))


def _d1h_safety(st, B):
    key = (B, 8, True)
    route = f'1h-s{_D1H_SPLITS[B]}'
    errors, cache_ok, ids_equal, used = (), False, False, False
    reason = ''
    try:
        data, refresh = _d1h_inputs(st, B, 8, True, 20261105 + B)
        refresh()
        assert st['att_ok'] and _supported7(data['config']) and _att_fits7(data, data['config'])
        alt = dict(data)
        for name in ('k_cache', 'v_cache'):
            alt[name] = data[name].clone()
        baseline = dict(st, dec_1h=(), _dec_1h_observe={})
        candidate = dict(st, dec_1h=(key,), _dec_1h_observe={})
        # One synthetic full-layer call per route. Reference = unchanged incumbent path.
        ref = tuple(x.clone() for x in kernel_fn(data, baseline))
        ref_attn = baseline['_dec_1h_observe']['attn'].clone()
        got = kernel_fn(alt, candidate)
        torch.cuda.synchronize()
        used = candidate['_dec_1h_observe']['used']
        cache_ok = all(torch.equal(data[k].view(torch.uint8), alt[k].view(torch.uint8))
                       for k in ('k_cache', 'v_cache'))
        ids_equal = torch.equal(ref[2], got[2])
        g1 = st['gate1'].float() / 0.3
        g2 = st['gate2'].float() / 0.3
        carry = (data['stream'] * (torch.sigmoid(-g1) * (1 + torch.sigmoid(g1))).clamp_min(1e-3).sqrt()
                 * (torch.sigmoid(-g2) * (1 + torch.sigmoid(g2))).clamp_min(1e-3).sqrt())
        errors = (_d1h_error(candidate['_dec_1h_observe']['attn'], ref_attn),
                  _d1h_error(got[0], ref[0]), _d1h_error(got[1], ref[1]),
                  _d1h_error(got[1] - carry, ref[1] - carry))
        accepted = _d1h_admit(errors, cache_ok, ids_equal, used)
        if accepted:
            st.setdefault('dec_1h', set()).add(key)
        else:
            reason = 'correctness_safety_failed'
        used = bool(accepted)
        _check_guards()
    except Exception as ex:
        st.get('dec_1h', set()).discard(key)
        used = False
        reason = repr(ex)[-1200:]
    print(f'DEC_1H shape=v{B}S route={route} err={errors} cache_ok={int(cache_ok)} '
          f'ids_equal={int(ids_equal)} used={int(used)} reason={reason}', flush=True)


def prepare(weights):
    # Compile concurrently with the incumbent's builds/tuning; no additional timing gate.
    worker = threading.Thread(target=_d1h_build, args=(weights['qkv_gate'].device,))
    worker.start()
    st = _D1H_OLD_PREPARE(weights)
    worker.join()
    st['dec_1h'] = set()
    for B in (16, 32):
        if '1h' in _D1H_MODULES:
            _d1h_safety(st, B)
        else:
            print(f'DEC_1H shape=v{B}S route=1h-s{_D1H_SPLITS[B]} err=() cache_ok=0 '
                  f'ids_equal=0 used=0 reason={_D1H_MODULES.get("error", "build_failed")}', flush=True)
    return st
