
# Dense S4 (DENSE_LEDGER.md): router (256 x 8192, FP32 weights) as a losslessly c12-packed BF16 hi + lo pair
# (v18s FMT 3, 24 bits per weight instead of 32) at M = 1, 8, 16, 32. TASK.md allows the hi + lo split with FP32
# accumulation at any m and lossless compression. v18s_plan proves the pack bit-exact, checks one-hot outputs
# against hi + lo exactly and FP64 random/spike rows at a quarter of the router tolerance. Each M is routed to a
# new geometry only if ABBA paired cold-L2 graph timing gives median ratio >= 1.03 with >= 75% of 16 pairs.
_S4_GEOMS = [(1, 4, 4, 8, 3), (1, 8, 4, 4, 3)]
_S4_MS = (1, 8, 16, 32)


def _s4_inst(mt, g):
    (wr, wk, ku, s, red) = g
    return (3, mt, 1, 2, red, 0, wr * wk * 32, 0, 256, 8192, wr, wk, ku, s, 6), (wr, wk, ku, s)


# --- runtime ---
import time as _s4_time


def _s4_graph(fn):
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, capture_error_mode='thread_local'):
        fn()
    return g


def _s4_once(g, flush):
    flush.sum()
    (e0, e1) = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
    e0.record()
    g.replay()
    e1.record()
    e1.synchronize()
    return e0.elapsed_time(e1) * 1000.0


def _s4_paired(ga, gb, flush, pairs=16):
    (ta, tb) = ([], [])
    for i in range(pairs):
        if i % 2 == 0:
            ta.append(_s4_once(ga, flush)); tb.append(_s4_once(gb, flush))
        else:
            tb.append(_s4_once(gb, flush)); ta.append(_s4_once(ga, flush))
    ratios = sorted(a / b for (a, b) in zip(ta, tb))
    return ratios[len(ratios) // 2], sum(r > 1.0 for r in ratios) / len(ratios), ta, tb


def _s4_select(state, weights):
    t0 = _s4_time.monotonic()
    w = weights['router']
    dev = w.device
    insts = list(dict.fromkeys(_s4_inst((m + 7) // 8, g)[0] for m in _S4_MS for g in _S4_GEOMS))
    ext = v18s_build(insts, 's4_v18s_router_fmt3')
    ids = {inst: i for (i, inst) in enumerate(insts)}
    print(f'S4_BUILD {_s4_time.monotonic() - t0:.0f}s', flush=True)
    flush = torch.ones(64 << 20, dtype=torch.float32, device=dev)
    gen = torch.Generator(device=dev).manual_seed(20261034)
    (packs, streams) = ({}, {})
    for m in _S4_MS:
        x = torch.randn((m, 8192), generator=gen, device=dev).to(torch.bfloat16)
        data = {'x': x, 'config': {'m': m, 'n': 256, 'k': 8192, 'layer': 'router'}}
        try:
            ref64 = x.double() @ w.double().T
            gbase = _s4_graph(lambda: _s4_prev_kernel(data, state))
        except Exception as ex:
            print(f'S4_SKIP m{m} baseline {type(ex).__name__}: {str(ex)[:120]}', flush=True)
            continue
        best = None
        for g in _S4_GEOMS:
            (inst, cfg) = _s4_inst((m + 7) // 8, g)
            try:
                plan = v18s_plan(ext, ids, packs, streams, w, 'router', ('router', 'm', m), inst, cfg)
                if plan is None:
                    print(f'S4_SKIP m{m} {g} plan/verify', flush=True)
                    continue
                out = torch.empty((m, 256), dtype=torch.float32, device=dev)
                run = lambda plan=plan, out=out, cfg=cfg: v18s_launch_raw(ext, plan['inst_id'], plan, x, out, 256, cfg)
                if run():
                    print(f'S4_SKIP m{m} {g} launch', flush=True)
                    continue
                torch.cuda.synchronize()
                err = float((out.double() - ref64).norm() / ref64.norm())
                gc = _s4_graph(run)
                (med, frac, ta, tb) = _s4_paired(gbase, gc, flush)
                del gc
            except Exception as ex:
                print(f'S4_SKIP m{m} {g} {type(ex).__name__}: {str(ex)[:160]}', flush=True)
                continue
            print(f'S4 m{m} {g} ratio {med:.4f} win {frac:.2f} relL2_vs_fp64 {err:.1e} esc {plan["max_esc"]} base_us '
                  + ','.join(f'{v:.1f}' for v in ta) + ' cand_us ' + ','.join(f'{v:.1f}' for v in tb), flush=True)
            if err < 2.5e-5 and med >= 1.03 and frac >= 0.75 and (best is None or med > best[0]):
                best = (med, g, plan, cfg)
        del gbase
        if best is not None:
            state['s4_sel'][m] = (ext, best[2], best[3])
            print(f'S4_SELECT m{m} {best[1]} {best[0]:.4f}', flush=True)
        else:
            print(f'S4_KEEP m{m}', flush=True)
    del flush
    torch.cuda.synchronize()
    print(f'S4_SELECTOR selected={sorted(state["s4_sel"])} s={_s4_time.monotonic() - t0:.1f}', flush=True)


_S4_SEEN = set()
_s4_prev_kernel = kernel_fn


def kernel_fn(data, state):
    sel = state.get('s4_sel')
    if sel:
        (x, cfg) = (data['x'], data['config'])
        if cfg['layer'] == 'router' and cfg['n'] == 256 and x.ndim == 2 and x.shape[1] == 8192:
            v = sel.get(x.shape[0])
            if v is not None and x.dtype == torch.bfloat16 and x.is_contiguous() and x.data_ptr() % 16 == 0 \
                    and x.device == v[1]['wp'].device:
                (ext, plan, c) = v
                out = torch.empty((x.shape[0], 256), dtype=torch.float32, device=x.device)
                if v18s_launch_raw(ext, plan['inst_id'], plan, x, out, 256, c) == 0:
                    if x.shape[0] not in _S4_SEEN:
                        _S4_SEEN.add(x.shape[0])
                        print(f'S4_USE m{x.shape[0]}', flush=True)
                    return out
    return _s4_prev_kernel(data, state)


_s4_prev_prepare = prepare


def prepare(weights):
    state = _s4_prev_prepare(weights)
    state['s4_sel'] = {}
    try:
        _s4_select(state, weights)
    except Exception as ex:
        state['s4_sel'] = {}
        print(f'S4_SELECTOR off {type(ex).__name__}: {ex}', flush=True)
    torch.cuda.synchronize()
    return state
