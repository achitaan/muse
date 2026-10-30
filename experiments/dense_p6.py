
# Dense P6 (PROGRESS.md H6): unpooler M32 (8192 x 4096) runs the m32w WGMMA kernel at ~17.0 us (kernel) while the
# same-size pooler M32 runs a GEMV (m32x) at ~16.3 us; the c12 byte floor is ~10.6 us. No GEMV has been tried for
# unpooler M32 (ledger: only m32w K-partition sweeps). v18d native c12 GEMV with MT=4 (m <= 32), several
# geometries; v18d_v18d_native_plan verification; ABBA gate (median >= 1.03, >= 75% of 16 pairs).
_P6_GEOMS = [((1, 4, 16, 1), 0), ((1, 8, 8, 1), 0), ((2, 4, 8, 1), 0), ((1, 4, 8, 2), 2), ((1, 8, 4, 2), 3)]


def _p6_inst(red):
    return (1, 4, 1, 1, red, 0, 256)


# --- runtime ---
def _p6_select(state, weights):
    t0 = _p1_time.monotonic()
    w = weights['unpooler']
    dev = w.device
    flush = torch.ones(64 << 20, dtype=torch.float32, device=dev)
    gen = torch.Generator(device=dev).manual_seed(20261044)
    x = torch.randn((32, 4096), generator=gen, device=dev).to(torch.bfloat16)
    data = {'x': x, 'config': {'m': 32, 'n': 8192, 'k': 4096, 'layer': 'unpooler'}}
    gbase = _p1_graph(lambda: _p6_prev_kernel(data, state))
    best = None
    for (g, red) in _P6_GEOMS:
        tag = f'unpooler m32 {g} RED{red}'
        try:
            st = v18d_prepare_small_m({'unpooler': w}, table_m={},
                                      table_mt={('unpooler', 4): (_p6_inst(red), g)},
                                      name='p6_unp32_' + '_'.join(map(str, g)) + f'_r{red}')
            if ('unpooler', 'mt', 4) not in st['plans']:
                print(f'P6_SKIP {tag} no verified plan', flush=True)
                continue
            if v18d_run_small_m(x, 'unpooler', 8192, st) is None:
                print(f'P6_SKIP {tag} launch', flush=True)
                continue
            gc = _p1_graph(lambda st=st: v18d_run_small_m(x, 'unpooler', 8192, st))
            (med, frac, ta, tb) = _p1_paired(gbase, gc, flush)
            del gc
        except Exception as ex:
            print(f'P6_SKIP {tag} {type(ex).__name__}: {str(ex)[:200]}', flush=True)
            continue
        print(f'P6 {tag} ratio {med:.4f} win {frac:.2f} base_us ' + ','.join(f'{v:.1f}' for v in ta)
              + ' cand_us ' + ','.join(f'{v:.1f}' for v in tb), flush=True)
        if med >= 1.03 and frac >= 0.75 and (best is None or med > best[0]):
            best = (med, st, tag)
    del gbase, flush
    if best is not None:
        state['p6_sel'][32] = best[1]
        print(f'P6_SELECT {best[2]} {best[0]:.4f}', flush=True)
    else:
        print('P6_KEEP unpooler m32', flush=True)
    torch.cuda.synchronize()
    print(f'P6_SELECTOR selected={sorted(state["p6_sel"])} s={_p1_time.monotonic() - t0:.1f}', flush=True)


_P6_SEEN = set()
_p6_prev_kernel = kernel_fn


def kernel_fn(data, state):
    sel = state.get('p6_sel')
    if sel and data['config']['layer'] == 'unpooler':
        x = data['x']
        st = sel.get(x.shape[0]) if x.ndim == 2 else None
        if st is not None:
            out = v18d_run_small_m(x, 'unpooler', data['config']['n'], st)
            if out is not None:
                if x.shape[0] not in _P6_SEEN:
                    _P6_SEEN.add(x.shape[0])
                    print(f'P6_USE unpooler m{x.shape[0]}', flush=True)
                return out
    return _p6_prev_kernel(data, state)


_p6_prev_prepare = prepare


def prepare(weights):
    state = _p6_prev_prepare(weights)
    state['p6_sel'] = {}
    try:
        _p6_select(state, weights)
    except Exception as ex:
        state['p6_sel'] = {}
        print(f'P6_SELECTOR off {type(ex).__name__}: {ex}', flush=True)
    torch.cuda.synchronize()
    return state
