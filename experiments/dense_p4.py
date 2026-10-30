
# Dense P4: confirmation of the two P2 near-misses in a fresh process, frozen to the winning configs only:
# o M8 v18s (4,8,2,1) NT1024 (P2: 1.025, 14/16) and router M16 routertail S16/WK4 (P2: 1.028, 16/16).
# Pre-registered rule (written before the check): route iff this run again shows median >= 1.015 and
# >= 75% paired wins, after the same correctness gates. Router M8 is not routed (P2 1.004).

# Dense P2 (PROGRESS.md H2): o M8 runs v18s (1,8,2,1) at 4.5 us kernel while o M1 runs 3.9 us with a 1024-thread
# (4,8,2,1) layout. Wider-CTA v18s layouts for o M8 on the same c12 kernel and pack (v18s_plan: bit-exact pack,
# one-hot, FP64). Rerouted only after ABBA paired cold-L2 graph timing (median >= 1.03, >= 75% of 16 pairs).
_P2_O_GEOMS = [((4, 8, 2, 1), 1024)]


def _p2_inst(g, nt):
    (wr, wk, ku, s) = g
    return (1, 1, 1, 2, 0, 0, nt, 0, 8192, 1024, wr, wk, ku, s, 6)


def _p2_o8(state, weights, flush, gen):
    w = weights['o']
    dev = w.device
    insts = [_p2_inst(g, nt) for (g, nt) in _P2_O_GEOMS]
    ext = v18s_build(insts, 'p2_v18s_o8')
    ids = {inst: i for (i, inst) in enumerate(insts)}
    (packs, streams) = ({}, {})
    x = torch.randn((8, 1024), generator=gen, device=dev).to(torch.bfloat16)
    data = {'x': x, 'config': {'m': 8, 'n': 8192, 'k': 1024, 'layer': 'o'}}
    gbase = _p1_graph(lambda: _p2_prev_kernel(data, state))
    best = None
    for (g, nt) in _P2_O_GEOMS:
        inst = _p2_inst(g, nt)
        tag = f'o m8 {g}'
        try:
            plan = v18s_plan(ext, ids, packs, streams, w, 'o', ('o', 'm', 8), inst, g)
            if plan is None:
                print(f'P4_SKIP {tag} plan/verify', flush=True)
                continue
            out = torch.empty((8, 8192), dtype=torch.bfloat16, device=dev)
            run = lambda plan=plan, out=out, g=g: v18s_launch_raw(ext, plan['inst_id'], plan, x, out, 8192, g)
            if run():
                print(f'P4_SKIP {tag} launch', flush=True)
                continue
            gc = _p1_graph(run)
            (med, frac, ta, tb) = _p1_paired(gbase, gc, flush)
            del gc
        except Exception as ex:
            print(f'P4_SKIP {tag} {type(ex).__name__}: {str(ex)[:160]}', flush=True)
            continue
        print(f'P4 {tag} ratio {med:.4f} win {frac:.2f} base_us ' + ','.join(f'{v:.1f}' for v in ta)
              + ' cand_us ' + ','.join(f'{v:.1f}' for v in tb), flush=True)
        if med >= 1.015 and frac >= 0.75 and (best is None or med > best[0]):
            best = (med, plan, g, tag)
    del gbase
    if best is not None:
        state['p2_sel'][8] = (ext, best[1], best[2])
        print(f'P4_SELECT {best[3]} {best[0]:.4f}', flush=True)
    else:
        print('P4_KEEP o m8', flush=True)


_P2_SEEN = set()
_p2_prev_kernel = kernel_fn


def kernel_fn(data, state):
    sel = state.get('p2_sel')
    if sel and data['config']['layer'] == 'o':
        x = data['x']
        v = sel.get(x.shape[0]) if x.ndim == 2 else None
        if v is not None and data['config']['n'] == 8192 and x.shape[1] == 1024 and x.dtype == torch.bfloat16 \
                and x.is_contiguous() and x.data_ptr() % 16 == 0 and x.device == v[1]['wp'].device:
            (ext, plan, g) = v
            out = torch.empty((x.shape[0], 8192), dtype=torch.bfloat16, device=x.device)
            if v18s_launch_raw(ext, plan['inst_id'], plan, x, out, 8192, g) == 0:
                if x.shape[0] not in _P2_SEEN:
                    _P2_SEEN.add(x.shape[0])
                    print(f'P4_USE o m{x.shape[0]}', flush=True)
                return out
    return _p2_prev_kernel(data, state)


_p2_prev_prepare = prepare


_P2_RT_ALTS = [(16, 4)]


def _p2_router(state, weights, flush, gen):
    w = weights['router']
    dev = w.device
    st = rt_prepare_routertail_candidates(weights)
    if not st.get('enabled'):
        print('P4_SKIP router routertail disabled', flush=True)
        return
    ext = st['ext']
    exact = (w.to(torch.bfloat16).float() + (w - w.to(torch.bfloat16).float()).to(torch.bfloat16).float())
    stream = lambda: torch.cuda.current_stream(dev).cuda_stream
    alts = {}
    for (s, wk) in _P2_RT_ALTS:
        alts[(s, wk)] = rt__pack(w, s, wk)
    cols = torch.randperm(8192, generator=torch.Generator().manual_seed(7))[:64].tolist()
    for m in (16,):
        x = torch.randn((m, 8192), generator=gen, device=dev).to(torch.bfloat16)
        data = {'x': x, 'config': {'m': m, 'n': 256, 'k': 8192, 'layer': 'router'}}
        gbase = _p1_graph(lambda: _p2_prev_kernel(data, state))
        xr = torch.randn((m, 8192), generator=gen, device=dev)
        xr[m - 1].zero_()
        xr[m - 1, 4321] = 16.0
        xr = xr.to(torch.bfloat16)
        ref = xr.double() @ w.double().T
        best = None
        for ((s, wk), packed) in alts.items():
            tag = f'router m{m} S{s}/WK{wk}'
            try:
                out = torch.empty((m, 256), dtype=torch.float32, device=dev)
                ok = ext.routertail_small(packed.data_ptr(), xr.data_ptr(), out.data_ptr(), m, s, wk, stream(), 0) == 0
                eye = torch.zeros((m, 8192), dtype=torch.bfloat16, device=dev)
                for c0 in range(0, len(cols), m):
                    eye.zero_()
                    cs = cols[c0:c0 + m]
                    for (r, c) in enumerate(cs):
                        eye[r, c] = 1.0
                    got = torch.empty((m, 256), dtype=torch.float32, device=dev)
                    ok = ok and ext.routertail_small(packed.data_ptr(), eye.data_ptr(), got.data_ptr(), m, s, wk, stream(), 0) == 0
                    ok = ok and bool(torch.equal(got[:len(cs)], exact[:, cs].t()))
                torch.cuda.synchronize()
                err = float((out.double() - ref).norm() / ref.norm())
                spike = float((out[m - 1].double() - ref[m - 1]).norm() / ref[m - 1].norm())
                if not ok or not bool(torch.isfinite(out).all()) or err > 2.5e-5 or spike > 2.5e-4:
                    print(f'P4_SKIP {tag} verify ok={ok} err={err:.1e} spike={spike:.1e}', flush=True)
                    continue
                outb = torch.empty((m, 256), dtype=torch.float32, device=dev)
                gc = _p1_graph(lambda packed=packed, s=s, wk=wk, outb=outb: ext.routertail_small(
                    packed.data_ptr(), x.data_ptr(), outb.data_ptr(), m, s, wk, stream(), 0))
                (med, frac, ta, tb) = _p1_paired(gbase, gc, flush)
                del gc
            except Exception as ex:
                print(f'P4_SKIP {tag} {type(ex).__name__}: {str(ex)[:160]}', flush=True)
                continue
            print(f'P4 {tag} ratio {med:.4f} win {frac:.2f} onehot_exact 1 relL2 {err:.1e} base_us ' + ','.join(f'{v:.1f}' for v in ta)
                  + ' cand_us ' + ','.join(f'{v:.1f}' for v in tb), flush=True)
            if med >= 1.015 and frac >= 0.75 and (best is None or med > best[0]):
                best = (med, (ext, packed, s, wk), tag)
        del gbase
        if best is not None:
            state['p2_rsel'][m] = best[1]
            print(f'P4_SELECT {best[2]} {best[0]:.4f}', flush=True)
        else:
            print(f'P4_KEEP router m{m}', flush=True)


_p2_prev_kernel2 = kernel_fn


def kernel_fn(data, state):
    sel = state.get('p2_rsel')
    if sel and data['config']['layer'] == 'router':
        x = data['x']
        v = sel.get(x.shape[0]) if x.ndim == 2 else None
        if v is not None and x.shape[1] == 8192 and x.dtype == torch.bfloat16 and x.is_contiguous() \
                and x.data_ptr() % 16 == 0 and x.device == v[1].device:
            (ext, packed, s, wk) = v
            out = torch.empty((x.shape[0], 256), dtype=torch.float32, device=x.device)
            if ext.routertail_small(packed.data_ptr(), x.data_ptr(), out.data_ptr(), x.shape[0], s, wk,
                                    torch.cuda.current_stream(x.device).cuda_stream, 0) == 0:
                if ('r', x.shape[0]) not in _P2_SEEN:
                    _P2_SEEN.add(('r', x.shape[0]))
                    print(f'P4_USE router m{x.shape[0]}', flush=True)
                return out
    return _p2_prev_kernel2(data, state)


def prepare(weights):
    state = _p2_prev_prepare(weights)
    state['p2_sel'] = {}
    state['p2_rsel'] = {}
    dev = weights['o'].device
    flush = torch.ones(64 << 20, dtype=torch.float32, device=dev)
    for part in (_p2_o8, _p2_router):
        try:
            part(state, weights, flush, torch.Generator(device=dev).manual_seed(20261041))
        except Exception as ex:
            print(f'P4_PART_OFF {part.__name__} {type(ex).__name__}: {str(ex)[:200]}', flush=True)
        torch.cuda.synchronize()
    del flush
    print(f'P4_SELECTOR o={sorted(state["p2_sel"])} router={sorted(state["p2_rsel"])}', flush=True)
    return state
