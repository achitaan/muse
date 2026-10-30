
# Dense P5 (PROGRESS.md H5): P3's ablation showed qkv M1's cluster reduction (RED 2: every warp of the S-1 non-owner
# CTAs pushes its partial through DSMEM) costs ~0.4 us (DIAG4 1.038). RED 3 first reduces inside each CTA, then
# pushes one partial per row group per CTA. Same v18d c12 kernel, pack and verification (v18d_v18d_native_plan);
# rerouted only on ABBA paired cold-L2 graph timing (median >= 1.03, >= 75% of 16 pairs) at M1 and M8.
_P5_GEOMS = [(1, 4, 8, 4), (1, 8, 4, 4), (1, 8, 8, 2)]


# --- runtime ---
def _p5_select(state, weights):
    t0 = _p1_time.monotonic()
    w = weights['qkv_gate']
    dev = w.device
    flush = torch.ones(64 << 20, dtype=torch.float32, device=dev)
    gen = torch.Generator(device=dev).manual_seed(20261043)
    xs = {m: torch.randn((m, 8192), generator=gen, device=dev).to(torch.bfloat16) for m in (1, 8)}
    gbase = {}
    for m in (1, 8):
        data = {'x': xs[m], 'config': {'m': m, 'n': 2304, 'k': 8192, 'layer': 'qkv_gate'}}
        gbase[m] = _p1_graph(lambda data=data: _p5_prev_kernel(data, state))
    for g in _P5_GEOMS:
        try:
            st = v18d_prepare_small_m({'qkv_gate': w}, table_m={},
                                      table_mt={('qkv_gate', 1): ((1, 1, 1, 2, 3, 0, 256), g)},
                                      name='p5_qkv_red3_' + '_'.join(map(str, g)))
        except Exception as ex:
            print(f'P5_SKIP build {g} {type(ex).__name__}: {str(ex)[:200]}', flush=True)
            continue
        if ('qkv_gate', 'mt', 1) not in st['plans']:
            print(f'P5_SKIP {g} no verified plan', flush=True)
            continue
        for m in (1, 8):
            tag = f'qkv m{m} RED3 {g}'
            try:
                if v18d_run_small_m(xs[m], 'qkv_gate', 2304, st) is None:
                    print(f'P5_SKIP {tag} launch', flush=True)
                    continue
                gc = _p1_graph(lambda st=st, m=m: v18d_run_small_m(xs[m], 'qkv_gate', 2304, st))
                (med, frac, ta, tb) = _p1_paired(gbase[m], gc, flush)
                del gc
            except Exception as ex:
                print(f'P5_SKIP {tag} {type(ex).__name__}: {str(ex)[:160]}', flush=True)
                continue
            print(f'P5 {tag} ratio {med:.4f} win {frac:.2f} base_us ' + ','.join(f'{v:.1f}' for v in ta)
                  + ' cand_us ' + ','.join(f'{v:.1f}' for v in tb), flush=True)
            if med >= 1.03 and frac >= 0.75 and (m not in state['p5_sel'] or med > state['p5_sel'][m][0]):
                state['p5_sel'][m] = (med, st, tag)
    for m in (1, 8):
        if m in state['p5_sel']:
            print(f'P5_SELECT {state["p5_sel"][m][2]} {state["p5_sel"][m][0]:.4f}', flush=True)
        else:
            print(f'P5_KEEP qkv m{m}', flush=True)
    del flush, gbase
    torch.cuda.synchronize()
    print(f'P5_SELECTOR selected={sorted(state["p5_sel"])} s={_p1_time.monotonic() - t0:.1f}', flush=True)


_P5_SEEN = set()
_p5_prev_kernel = kernel_fn


def kernel_fn(data, state):
    sel = state.get('p5_sel')
    if sel and data['config']['layer'] == 'qkv_gate':
        x = data['x']
        v = sel.get(x.shape[0]) if x.ndim == 2 else None
        if v is not None:
            out = v18d_run_small_m(x, 'qkv_gate', data['config']['n'], v[1])
            if out is not None:
                if x.shape[0] not in _P5_SEEN:
                    _P5_SEEN.add(x.shape[0])
                    print(f'P5_USE qkv m{x.shape[0]}', flush=True)
                return out
    return _p5_prev_kernel(data, state)


_p5_prev_prepare = prepare


def prepare(weights):
    state = _p5_prev_prepare(weights)
    state['p5_sel'] = {}
    try:
        _p5_select(state, weights)
    except Exception as ex:
        state['p5_sel'] = {}
        print(f'P5_SELECTOR off {type(ex).__name__}: {ex}', flush=True)
    torch.cuda.synchronize()
    return state
