
# Dense P1 (PROGRESS.md H1): qkv_gate M1/M8/M16 wave balance. qkv has 2304 rows = 144 16-row tiles; the incumbent
# v18d (1,4,8,4) launches 576 CTAs (4.36 per SM on 132 SMs). Geometries with S=8 (1152 CTAs, 8.73 per SM) or more
# warps per CTA are built on the same v18d native c12 kernel and pack (v18d_v18d_native_plan: bit-exact pack,
# one-hot, FP64 random). Each shape is rerouted only after ABBA paired cold-L2 graph timing vs the incumbent
# (median >= 1.03, >= 75% of 16 pairs).
_P1_GEOMS = [((1, 4, 4, 8), 2, 256), ((1, 8, 2, 8), 2, 256), ((1, 2, 8, 8), 2, 256),
             ((1, 8, 4, 4), 2, 256), ((1, 16, 8, 1), 0, 512), ((2, 4, 4, 8), 2, 256), ((1, 4, 2, 16), 2, 256)]
_P1_MS = (1, 8, 16)


def _p1_insts(g, red, nt):
    # (FMT, MT, RT, ST, RED, 0, NT) as in v18d_SM_TABLE_MT['qkv_gate', 1] / ['qkv_gate', 2]
    return {1: (1, 1, 1, 2, red, 0, nt), 2: (1, 2, 1, 1, red, 0, nt)}


# --- runtime ---
import time as _p1_time


def _p1_graph(fn):
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, capture_error_mode='thread_local'):
        fn()
    return g


def _p1_once(g, flush):
    flush.sum()
    (e0, e1) = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
    e0.record()
    g.replay()
    e1.record()
    e1.synchronize()
    return e0.elapsed_time(e1) * 1000.0


def _p1_paired(ga, gb, flush, pairs=16):
    (ta, tb) = ([], [])
    for i in range(pairs):
        if i % 2 == 0:
            ta.append(_p1_once(ga, flush)); tb.append(_p1_once(gb, flush))
        else:
            tb.append(_p1_once(gb, flush)); ta.append(_p1_once(ga, flush))
    ratios = sorted(a / b for (a, b) in zip(ta, tb))
    return ratios[len(ratios) // 2], sum(r > 1.0 for r in ratios) / len(ratios), ta, tb


def _p1_select(state, weights):
    t0 = _p1_time.monotonic()
    w = weights['qkv_gate']
    dev = w.device
    built = []
    for (i, (g, red, nt)) in enumerate(_P1_GEOMS):
        ins = _p1_insts(g, red, nt)
        try:
            st = v18d_prepare_small_m({'qkv_gate': w}, table_m={},
                                      table_mt={('qkv_gate', 1): (ins[1], g), ('qkv_gate', 2): (ins[2], g)},
                                      name=f'p1_qkv_g{i}_' + '_'.join(map(str, g)))
            print(f'P1_BUILD {g} plans={sorted(k[2] for k in st["plans"])} t={_p1_time.monotonic() - t0:.0f}s', flush=True)
            built.append((g, st))
        except Exception as ex:
            print(f'P1_SKIP build {g} {type(ex).__name__}: {str(ex)[:200]}', flush=True)
        if _p1_time.monotonic() - t0 > 150:
            print('P1_BUILD_BUDGET stop', flush=True)
            break
    flush = torch.ones(64 << 20, dtype=torch.float32, device=dev)
    gen = torch.Generator(device=dev).manual_seed(20261040)
    for m in _P1_MS:
        x = torch.randn((m, 8192), generator=gen, device=dev).to(torch.bfloat16)
        data = {'x': x, 'config': {'m': m, 'n': 2304, 'k': 8192, 'layer': 'qkv_gate'}}
        try:
            gbase = _p1_graph(lambda: _p1_prev_kernel(data, state))
        except Exception as ex:
            print(f'P1_SKIP m{m} baseline {type(ex).__name__}: {str(ex)[:120]}', flush=True)
            continue
        best = None
        for (g, st) in built:
            tag = f'qkv m{m} {g}'
            if v18d_v18d_native_mt(m) not in [k[2] for k in st['plans']]:
                print(f'P1_SKIP {tag} no verified plan', flush=True)
                continue
            try:
                if v18d_run_small_m(x, 'qkv_gate', 2304, st) is None:
                    print(f'P1_SKIP {tag} launch', flush=True)
                    continue
                gc = _p1_graph(lambda st=st: v18d_run_small_m(x, 'qkv_gate', 2304, st))
                (med, frac, ta, tb) = _p1_paired(gbase, gc, flush)
                del gc
            except Exception as ex:
                print(f'P1_SKIP {tag} {type(ex).__name__}: {str(ex)[:160]}', flush=True)
                continue
            print(f'P1 {tag} ratio {med:.4f} win {frac:.2f} base_us ' + ','.join(f'{v:.1f}' for v in ta)
                  + ' cand_us ' + ','.join(f'{v:.1f}' for v in tb), flush=True)
            if med >= 1.03 and frac >= 0.75 and (best is None or med > best[0]):
                best = (med, st, tag)
        del gbase
        if best is not None:
            state['p1_sel'][m] = best[1]
            print(f'P1_SELECT {best[2]} {best[0]:.4f}', flush=True)
        else:
            print(f'P1_KEEP qkv m{m}', flush=True)
    del flush
    torch.cuda.synchronize()
    print(f'P1_SELECTOR selected={sorted(state["p1_sel"])} s={_p1_time.monotonic() - t0:.1f}', flush=True)


_P1_SEEN = set()
_p1_prev_kernel = kernel_fn


def kernel_fn(data, state):
    sel = state.get('p1_sel')
    if sel and data['config']['layer'] == 'qkv_gate':
        x = data['x']
        st = sel.get(x.shape[0]) if x.ndim == 2 else None
        if st is not None:
            out = v18d_run_small_m(x, 'qkv_gate', data['config']['n'], st)
            if out is not None:
                if x.shape[0] not in _P1_SEEN:
                    _P1_SEEN.add(x.shape[0])
                    print(f'P1_USE qkv m{x.shape[0]}', flush=True)
                return out
    return _p1_prev_kernel(data, state)


_p1_prev_prepare = prepare


def prepare(weights):
    state = _p1_prev_prepare(weights)
    state['p1_sel'] = {}
    try:
        _p1_select(state, weights)
    except Exception as ex:
        state['p1_sel'] = {}
        print(f'P1_SELECTOR off {type(ex).__name__}: {ex}', flush=True)
    torch.cuda.synchronize()
    return state
