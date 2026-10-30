
# Dense P3 (PROGRESS.md H3, oracle #4 diagnostic + one real knob). qkv_gate M1 runs v18d (1,4,8,4) at ~3.0 TB/s
# effective vs pooler 3.7 TB/s, and P1 showed layout is not the cause.
# (a) Diagnostic only (never routed): the same kernel with DIAG ablations, timed against the incumbent:
#     1 no MMA, 2 no x loads, 4 no cluster reduction, 8 weight loads only (no decode/MMA), 16 empty kernel.
# (b) Candidate: the incumbent geometry with more loads in flight, ST 3 / 4 (incumbent ST 2), verified by
#     v18d_v18d_native_plan and gated by ABBA (median >= 1.03, >= 75% of 16 pairs) at M1 and M8.
_P3_CFG = (1, 4, 8, 4)
_P3_DIAGS = (1, 2, 4, 8, 16)
_P3_STS = (3, 4)


def _p3_diag_inst(d):
    return (1, 1, 1, 2, 2, d, 256, 0, 2304, 8192) + _P3_CFG


# --- runtime ---
def _p3_select(state, weights):
    t0 = _p1_time.monotonic()
    w = weights['qkv_gate']
    dev = w.device
    flush = torch.ones(64 << 20, dtype=torch.float32, device=dev)
    gen = torch.Generator(device=dev).manual_seed(20261042)
    xs = {m: torch.randn((m, 8192), generator=gen, device=dev).to(torch.bfloat16) for m in (1, 8)}
    gbase = {}
    for m in (1, 8):
        data = {'x': xs[m], 'config': {'m': m, 'n': 2304, 'k': 8192, 'layer': 'qkv_gate'}}
        gbase[m] = _p1_graph(lambda data=data: _p3_prev_kernel(data, state))
    try:
        insts = [_p3_diag_inst(d) for d in _P3_DIAGS]
        ext = v18d_v18d_native_build(insts, 'p3_qkv1_diag')
        pk = v18d_v18d_native_pack_format(w, 1)
        sp = v18d_v18d_native_stream(pk, 1, *_P3_CFG)
        print(f'P3_DIAG_BUILD t={_p1_time.monotonic() - t0:.0f}s stream={sp is not None}', flush=True)
        out = torch.empty((1, 2304), dtype=torch.bfloat16, device=dev)
        for (i, d) in enumerate(_P3_DIAGS):
            run = lambda i=i: v18d_v18d_native_launch_raw(ext, i, sp, xs[1], out, 2304, _P3_CFG)
            rc = run()
            torch.cuda.synchronize()
            if rc:
                print(f'P3_DIAG {d} rc {rc}', flush=True)
                continue
            gc = _p1_graph(run)
            (med, frac, ta, tb) = _p1_paired(gbase[1], gc, flush)
            del gc
            print(f'P3_DIAG {d} m1 ratio {med:.4f} base_med {sorted(ta)[8]:.2f} diag_med {sorted(tb)[8]:.2f} base_us '
                  + ','.join(f'{v:.1f}' for v in ta) + ' diag_us ' + ','.join(f'{v:.1f}' for v in tb), flush=True)
        del ext, pk, sp
    except Exception as ex:
        print(f'P3_DIAG off {type(ex).__name__}: {str(ex)[:200]}', flush=True)
    torch.cuda.synchronize()
    for st_n in _P3_STS:
        try:
            st = v18d_prepare_small_m({'qkv_gate': w}, table_m={},
                                      table_mt={('qkv_gate', 1): ((1, 1, 1, st_n, 2, 0, 256), _P3_CFG)},
                                      name=f'p3_qkv1_st{st_n}')
        except Exception as ex:
            print(f'P3_SKIP build ST{st_n} {type(ex).__name__}: {str(ex)[:200]}', flush=True)
            continue
        if ('qkv_gate', 'mt', 1) not in st['plans']:
            print(f'P3_SKIP ST{st_n} no verified plan', flush=True)
            continue
        for m in (1, 8):
            tag = f'qkv m{m} ST{st_n}'
            try:
                gc = _p1_graph(lambda st=st, m=m: v18d_run_small_m(xs[m], 'qkv_gate', 2304, st))
                (med, frac, ta, tb) = _p1_paired(gbase[m], gc, flush)
                del gc
            except Exception as ex:
                print(f'P3_SKIP {tag} {type(ex).__name__}: {str(ex)[:160]}', flush=True)
                continue
            print(f'P3 {tag} ratio {med:.4f} win {frac:.2f} base_us ' + ','.join(f'{v:.1f}' for v in ta)
                  + ' cand_us ' + ','.join(f'{v:.1f}' for v in tb), flush=True)
            if med >= 1.03 and frac >= 0.75 and (m not in state['p3_sel'] or med > state['p3_sel'][m][0]):
                state['p3_sel'][m] = (med, st, tag)
    for m in (1, 8):
        if m in state['p3_sel']:
            print(f'P3_SELECT {state["p3_sel"][m][2]} {state["p3_sel"][m][0]:.4f}', flush=True)
        else:
            print(f'P3_KEEP qkv m{m}', flush=True)
    del flush, gbase
    torch.cuda.synchronize()
    print(f'P3_SELECTOR selected={sorted(state["p3_sel"])} s={_p1_time.monotonic() - t0:.1f}', flush=True)


_P3_SEEN = set()
_p3_prev_kernel = kernel_fn


def kernel_fn(data, state):
    sel = state.get('p3_sel')
    if sel and data['config']['layer'] == 'qkv_gate':
        x = data['x']
        v = sel.get(x.shape[0]) if x.ndim == 2 else None
        if v is not None:
            out = v18d_run_small_m(x, 'qkv_gate', data['config']['n'], v[1])
            if out is not None:
                if x.shape[0] not in _P3_SEEN:
                    _P3_SEEN.add(x.shape[0])
                    print(f'P3_USE qkv m{x.shape[0]}', flush=True)
                return out
    return _p3_prev_kernel(data, state)


_p3_prev_prepare = prepare


def prepare(weights):
    state = _p3_prev_prepare(weights)
    state['p3_sel'] = {}
    try:
        _p3_select(state, weights)
    except Exception as ex:
        state['p3_sel'] = {}
        print(f'P3_SELECTOR off {type(ex).__name__}: {ex}', flush=True)
    torch.cuda.synchronize()
    return state
