
# Dense S1 (branch omos/dense-top50-20261030, DENSE_LEDGER.md Check 1, narrowed per review):
# sliced unpooler (1024 x 4096, exact c12) at M = 1 and 8 only. Base = best/verified.py without the
# dead _achita_wg prepare sweep. Two v18s_gemv geometries are compared with the incumbent dispatch:
#   RED3: S=2 cluster, 8 warps x 4 K-chunks, intra-CTA reduction then one DSMEM push per CTA
#         (incumbent: S=4/S=2 clusters, every warp pushes, 2 K-chunks per warp);
#   RED0: no cluster, 16 warps x 4 K-chunks, one shared-memory reduction.
# Packing is proven lossless (v18s_pack_format unpack == original bits); v18s_plan checks one-hot
# exactness and FP64 random/spike rows. Timing is balanced ABBA pairs of cold-L2 CUDA-graph replays;
# a geometry is used only if the median paired ratio is >= 1.03 and >= 75% of pairs favour it.
import time as _s1_time

_S1_GEOMS = [(1, 8, 4, 2, 3), (1, 16, 4, 1, 0)]
_S1_MS = (1, 8)


def _s1_inst(g):
    (wr, wk, ku, s, red) = g
    return (1, 1, 1, 2, red, 0, wr * wk * 32, 0, 1024, 4096, wr, wk, ku, s, 6), (wr, wk, ku, s)


def _s1_graph(fn):
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    return g


def _s1_once(g, flush):
    flush.sum()
    (e0, e1) = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
    e0.record()
    g.replay()
    e1.record()
    e1.synchronize()
    return e0.elapsed_time(e1) * 1000.0


def _s1_paired(ga, gb, flush, pairs=16):
    (ta, tb) = ([], [])
    for i in range(pairs):
        if i % 2 == 0:
            ta.append(_s1_once(ga, flush)); tb.append(_s1_once(gb, flush))
        else:
            tb.append(_s1_once(gb, flush)); ta.append(_s1_once(ga, flush))
    ratios = sorted(a / b for (a, b) in zip(ta, tb))
    return ratios[len(ratios) // 2], sum(r > 1.0 for r in ratios) / len(ratios), ta, tb


def _s1_select(state, weights):
    t0 = _s1_time.monotonic()
    dev = weights['unpooler'].device
    insts = [_s1_inst(g)[0] for g in _S1_GEOMS]
    ext = v18s_build(insts, name='s1_v18s_unpooler')
    ids = {inst: i for (i, inst) in enumerate(insts)}
    print(f'S1_BUILD {_s1_time.monotonic() - t0:.0f}s', flush=True)
    flush = torch.ones(64 << 20, dtype=torch.float32, device=dev)
    gen = torch.Generator(device=dev).manual_seed(20261030)
    (w, n, k) = (weights['unpooler'], 1024, 4096)
    (packs, streams) = ({}, {})
    for m in _S1_MS:
        x = torch.randn((m, k), generator=gen, device=dev).to(torch.bfloat16)
        data = {'x': x, 'config': {'m': m, 'n': n, 'k': k, 'layer': 'unpooler'}}
        gbase = _s1_graph(lambda: _s1_prev_kernel(data, state))
        ref = _s1_prev_kernel(data, state).double()
        best = None
        for g in _S1_GEOMS:
            (inst, cfg) = _s1_inst(g)
            plan = v18s_plan(ext, ids, packs, streams, w, 'unpooler', ('unpooler', 'm', m), inst, cfg)
            if plan is None:
                print(f'S1_SKIP m{m} {g} plan/verify', flush=True)
                continue
            print(f'S1_PACK m{m} {g} rows {plan["rows"]} esc {packs[("unpooler", plan["rows"], 1)].get("n_esc")} max_esc_warp {plan["max_esc"]} bitexact 1', flush=True)
            out = torch.empty((m, n), dtype=torch.bfloat16, device=dev)
            run = lambda: v18s_launch_raw(ext, plan['inst_id'], plan, x, out, n, plan['cfg'])
            if run() != 0:
                print(f'S1_SKIP m{m} {g} launch', flush=True)
                continue
            agree = float((out.double() - ref).norm() / ref.norm().clamp_min(1e-30))
            gc = _s1_graph(run)
            (med, frac, ta, tb) = _s1_paired(gbase, gc, flush)
            print(f'S1 m{m} {g} ratio {med:.4f} win {frac:.2f} vs_incumbent {agree:.1e} base_us ' + ','.join(f'{v:.1f}' for v in ta)
                  + ' cand_us ' + ','.join(f'{v:.1f}' for v in tb), flush=True)
            del gc
            if med >= 1.03 and frac >= 0.75 and (best is None or med > best[0]):
                best = (med, g, plan)
        del gbase
        if best is not None:
            state['s1_sel'][(m, 'unpooler', n)] = (ext, best[2], n)
            print(f'S1_SELECT m{m} {best[1]} {best[0]:.4f}', flush=True)
        else:
            print(f'S1_KEEP m{m}', flush=True)
    del flush
    torch.cuda.synchronize()
    print(f'S1_SELECTOR selected={sorted(m for (m, _, _) in state["s1_sel"])} s={_s1_time.monotonic() - t0:.1f}', flush=True)


_S1_SEEN = set()
_s1_prev_kernel = kernel_fn


def kernel_fn(data, state):
    sel = state.get('s1_sel')
    if sel:
        (x, cfg) = (data['x'], data['config'])
        key = (x.shape[0], cfg['layer'], cfg['n'])
        v = sel.get(key)
        if v is not None and x.ndim == 2 and x.shape[1] == 4096 and x.dtype == torch.bfloat16 and x.is_contiguous() \
                and x.data_ptr() % 16 == 0 and x.device == v[1]['wp'].device:
            (ext, plan, n) = v
            out = torch.empty((x.shape[0], n), dtype=torch.bfloat16, device=x.device)
            if v18s_launch_raw(ext, plan['inst_id'], plan, x, out, n, plan['cfg']) == 0:
                if key not in _S1_SEEN:
                    _S1_SEEN.add(key)
                    print(f'S1_USE {key}', flush=True)
                return out
    return _s1_prev_kernel(data, state)


_s1_prev_prepare = prepare


def prepare(weights):
    state = _s1_prev_prepare(weights)
    state['s1_sel'] = {}
    try:
        _s1_select(state, weights)
    except Exception as ex:
        state['s1_sel'] = {}
        print(f'S1_SELECTOR off {type(ex).__name__}: {ex}', flush=True)
    torch.cuda.synchronize()
    return state
