
# Dense S6 (DENSE_LEDGER.md, oracle plan 3d #2 in its effective form): at M=32 each GEMV CTA covers only 16 rows,
# so x (32 x K) is re-read by every row group. v18s RT=2 gives each warp two 16-row tiles that reuse the same
# x registers: 32 rows per CTA halves x traffic (pooler 128 -> 64 MB, qkv 74 -> 37 MB) with the incumbent K split.
# Exactness via v18s_plan (bit-exact pack, one-hot, FP64); ABBA paired cold-L2 graph gate (median >= 1.03,
# >= 75% of 16 pairs).
_S6_CASES = {
    'pooler': (4096, 8192, (1, 8, 16, 1), 0),
    'qkv_gate': (2304, 8192, (1, 4, 8, 4), 2),
}
_S6_STS = (2, 1)


def _s6_inst(layer, st):
    (n, k, (wr, wk, ku, s), red) = _S6_CASES[layer]
    return (1, 4, 2, st, red, 0, wr * wk * 32, 0, n, k, wr, wk, ku, s, 6), (wr, wk, ku, s)


# --- runtime ---
import time as _s6_time


def _s6_graph(fn):
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, capture_error_mode='thread_local'):
        fn()
    return g


def _s6_once(g, flush):
    flush.sum()
    (e0, e1) = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
    e0.record()
    g.replay()
    e1.record()
    e1.synchronize()
    return e0.elapsed_time(e1) * 1000.0


def _s6_paired(ga, gb, flush, pairs=16):
    (ta, tb) = ([], [])
    for i in range(pairs):
        if i % 2 == 0:
            ta.append(_s6_once(ga, flush)); tb.append(_s6_once(gb, flush))
        else:
            tb.append(_s6_once(gb, flush)); ta.append(_s6_once(ga, flush))
    ratios = sorted(a / b for (a, b) in zip(ta, tb))
    return ratios[len(ratios) // 2], sum(r > 1.0 for r in ratios) / len(ratios), ta, tb


def _s6_select(state, weights):
    t0 = _s6_time.monotonic()
    insts = [_s6_inst(l, st)[0] for l in _S6_CASES for st in _S6_STS]
    ext = v18s_build(insts, 's6_v18s_rt2_m32')
    ids = {inst: i for (i, inst) in enumerate(insts)}
    print(f'S6_BUILD {_s6_time.monotonic() - t0:.0f}s', flush=True)
    dev = weights['pooler'].device
    flush = torch.ones(64 << 20, dtype=torch.float32, device=dev)
    gen = torch.Generator(device=dev).manual_seed(20261036)
    (packs, streams) = ({}, {})
    for layer in _S6_CASES:
        (n, k, _, _) = _S6_CASES[layer]
        x = torch.randn((32, k), generator=gen, device=dev).to(torch.bfloat16)
        data = {'x': x, 'config': {'m': 32, 'n': n, 'k': k, 'layer': layer}}
        try:
            gbase = _s6_graph(lambda: _s6_prev_kernel(data, state))
        except Exception as ex:
            print(f'S6_SKIP {layer} baseline {type(ex).__name__}: {str(ex)[:120]}', flush=True)
            continue
        best = None
        for st in _S6_STS:
            (inst, cfg) = _s6_inst(layer, st)
            tag = f'{layer} m32 RT2 ST{st} {cfg}'
            try:
                plan = v18s_plan(ext, ids, packs, streams, weights[layer], layer, (layer, 'm', 32), inst, cfg)
                if plan is None:
                    print(f'S6_SKIP {tag} plan/verify', flush=True)
                    continue
                out = torch.empty((32, n), dtype=torch.bfloat16, device=dev)
                run = lambda plan=plan, out=out, cfg=cfg: v18s_launch_raw(ext, plan['inst_id'], plan, x, out, n, cfg)
                if run():
                    print(f'S6_SKIP {tag} launch', flush=True)
                    continue
                gc = _s6_graph(run)
                (med, frac, ta, tb) = _s6_paired(gbase, gc, flush)
                del gc
            except Exception as ex:
                print(f'S6_SKIP {tag} {type(ex).__name__}: {str(ex)[:160]}', flush=True)
                continue
            print(f'S6 {tag} ratio {med:.4f} win {frac:.2f} max_esc {plan["max_esc"]} base_us ' + ','.join(f'{v:.1f}' for v in ta)
                  + ' cand_us ' + ','.join(f'{v:.1f}' for v in tb), flush=True)
            if med >= 1.03 and frac >= 0.75 and (best is None or med > best[0]):
                best = (med, plan, cfg, tag)
        del gbase
        if best is not None:
            state['s6_sel'][(32, layer, n)] = (ext, best[1], best[2])
            print(f'S6_SELECT {best[3]} {best[0]:.4f}', flush=True)
        else:
            print(f'S6_KEEP {layer} m32', flush=True)
    del flush
    torch.cuda.synchronize()
    print(f'S6_SELECTOR selected={sorted(state["s6_sel"])} s={_s6_time.monotonic() - t0:.1f}', flush=True)


_S6_SEEN = set()
_s6_prev_kernel = kernel_fn


def kernel_fn(data, state):
    sel = state.get('s6_sel')
    if sel:
        (x, cfg) = (data['x'], data['config'])
        if x.ndim == 2:
            key = (x.shape[0], cfg['layer'], cfg['n'])
            v = sel.get(key)
            if v is not None and x.shape[1] == 8192 and x.dtype == torch.bfloat16 and x.is_contiguous() \
                    and x.data_ptr() % 16 == 0 and x.device == v[1]['wp'].device:
                (ext, plan, c) = v
                out = torch.empty((x.shape[0], cfg['n']), dtype=torch.bfloat16, device=x.device)
                if v18s_launch_raw(ext, plan['inst_id'], plan, x, out, cfg['n'], c) == 0:
                    if key not in _S6_SEEN:
                        _S6_SEEN.add(key)
                        print(f'S6_USE {key}', flush=True)
                    return out
    return _s6_prev_kernel(data, state)


_s6_prev_prepare = prepare


def prepare(weights):
    state = _s6_prev_prepare(weights)
    state['s6_sel'] = {}
    try:
        _s6_select(state, weights)
    except Exception as ex:
        state['s6_sel'] = {}
        print(f'S6_SELECTOR off {type(ex).__name__}: {ex}', flush=True)
    torch.cuda.synchronize()
    return state
