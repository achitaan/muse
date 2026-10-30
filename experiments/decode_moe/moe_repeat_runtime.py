"""Prepare-only localization of TR8 byte failures; timed kernel stays parent."""

_mr_parent_prepare = prepare
_mr_parent_kernel = kernel_fn


def _mr_snapshot(state, output, n):
    # Each M<=32 call is one native launch. ctr[3] advances at its global
    # barrier, so the completed launch used (new_generation-1)&1.
    gen = int(state['scratch'][0][3].item())
    bank = (gen - 1) & 1
    ss = (state['scratch'][1][:n] if bank == 0 else
          state['scratch'][0][1032:1032+n].view(torch.float32))
    return dict(generation=gen, bank=bank, out=output.clone(), ss=ss.clone(),
        inter=state['scratch'][2][:n*HIDDEN].clone(),
        ybuf=state['scratch'][1][EMAX:EMAX+n*HIDDEN].clone())


def _mr_difference(a, b):
    report = {}
    for key in ['inter','ybuf','ss','out']:
        x, y = a[key], b[key]
        assert x.shape == y.shape and x.dtype == y.dtype
        mask = x.view(torch.uint8) != y.view(torch.uint8)
        count = int(mask.sum())
        finite = bool(torch.isfinite(x).all()) and bool(torch.isfinite(y).all())
        d = dict(byte_mismatches=count, finite=finite)
        if count:
            xf, yf = x.float().flatten(), y.float().flatten()
            changed = (x.flatten().view(torch.int16 if x.dtype == torch.bfloat16 else torch.int32)
                       != y.flatten().view(torch.int16 if y.dtype == torch.bfloat16 else torch.int32))
            idx = torch.nonzero(changed, as_tuple=False).flatten()[:6]
            xv, yv = xf[idx].cpu().tolist(), yf[idx].cpu().tolist()
            if not finite:
                xv, yv = list(map(str, xv)), list(map(str, yv))
            d.update(changed_elements=int(changed.sum()), samples=[idx.cpu().tolist(), xv, yv])
            if finite:
                delta = xf-yf
                d['max_abs'] = float(delta.abs().max())
                d['relative_L2'] = float(torch.linalg.vector_norm(delta) /
                    torch.linalg.vector_norm(xf).clamp_min(1e-30))
        report[key] = d
    return report


def _mr_case(m, state, other, variants, dev, start):
    data = _tr_probe_data(m, state, dev)
    _tr_mutate(data, state, 'one', 0)
    choices = [('A', lambda: _mr_parent_kernel(data, state), state),
               ('B', lambda: _mr_parent_kernel(data, state), state),
               ('C', lambda: _mr_parent_kernel(data, other), other)]
    if m <= 8 and ('tiny' if m <= 4 else 'tiny2') in variants:
        ext = variants['tiny' if m <= 4 else 'tiny2']
        variant_state = dict(other, tr8={m: ext})
        choices.append(('TR8', lambda: _mr_variant_launch(data, variant_state), variant_state))
    else:
        choices.append(('D', lambda: _mr_parent_kernel(data, other), other))
    last = choices[-1][0]
    graphs = {}
    for name, fn, st in choices:
        fn(); torch.cuda.synchronize(dev)
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            output = fn()
        graphs[name] = gr, output, st
    for pattern in ['none','one','four','all','skew']:
        for seed in range(3):
            if not _okh_budget(start, 278.):
                print('MR_BUDGET_STOP', m, pattern, seed, flush=True)
                return
            _tr_mutate(data, state, pattern, seed)
            # Diagnostic only: count valid rows in fresh synthetic inputs.
            # This host read never enters kernel_fn or a timed graph.
            ids = data['topk_ids']
            n = int(((ids >= state['lo']) & (ids < state['lo'] + state['cnt'])).sum())
            snapshots = []
            for cycle in range(2):
                snap = {}
                for name, graph_name in [('A','A'),('A_next','A'),('B','B'),
                                          ('C','C'),('C_next','C'),(last,last)]:
                    gr, output, st = graphs[graph_name]
                    before = int(st['scratch'][0][3].item())
                    gr.replay(); torch.cuda.synchronize(dev)
                    snap[name] = _mr_snapshot(st, output, n)
                    assert snap[name]['generation'] == before + 1
                snapshots.append(snap)
            first, second = snapshots
            comparisons = {
                'A_consecutive': _mr_difference(first['A'], first['A_next']),
                'C_consecutive': _mr_difference(first['C'], first['C_next']),
                'A_repeat': _mr_difference(first['A'], second['A']),
                'same_scratch_graph': _mr_difference(first['A'], first['B']),
                'independent_scratch': _mr_difference(first['A'], first['C'])}
            if 'TR8' in first:
                comparisons['TR8_vs_same_scratch_parent'] = _mr_difference(first['C'], first['TR8'])
                comparisons['TR8_repeat'] = _mr_difference(first['TR8'], second['TR8'])
            else:
                comparisons['independent_same_scratch_graph'] = _mr_difference(first['C'], first['D'])
            if pattern == 'none':
                assert all(bool((s['out'] == 0).all()) for snap in snapshots for s in snap.values())
            payload = dict(m=m, pattern=pattern, seed=seed, local_routes=n,
                generations={k:[first[k]['generation'],second[k]['generation']] for k in first},
                banks={k:[first[k]['bank'],second[k]['bank']] for k in first},
                comparisons=comparisons)
            print('MR_CASE', __import__('json').dumps(payload, allow_nan=False), flush=True)


def prepare(weights):
    start = _okh_clock.perf_counter()
    dev = weights['w13'].device
    lo, cnt = LOCAL_START, LOCAL_EXPERTS
    sfx = ('-DOKC_PRMTSC=1',) if (PRMT_SCALES and
        _scales_prmt_exact(weights['w13_sf'][lo:lo+cnt], weights['w2_sf'][lo:lo+cnt])) else ()
    built = {}
    def build():
        try:
            with torch.cuda.device(dev):
                for key, defs in [('tiny', TINY_DEFINES), ('tiny2', TINY2_DEFINES)]:
                    if key == 'tiny2' and not sfx and '-DOKC_WGS=1' in defs:
                        continue
                    built[key] = _mr_variant_ext(tuple(defs) + sfx + ('-DOKC_ROUTE_WARP=1',))
        except Exception as e:
            print('MR_VARIANT_BUILD_ERROR', repr(e)[-1500:], flush=True)
    thread = _okh_bg(build)
    state = _mr_parent_prepare(weights)
    thread.join()
    try:
        assert tuple(state.get('sfx', ())) == sfx
        other = dict(state, scratch=_make_scratch(dev))
        print('MR_START diagnostic_only=True timed_path=33114871 variants=', sorted(built),
              'grid=', state['grid'], 'sfx=', state.get('sfx', ()), flush=True)
        for m in [1, 3, 8, 16, 32]:
            if not _okh_budget(start, 278.):
                break
            _mr_case(m, state, other, built, dev, start)
    except Exception as e:
        print('MR_ERROR', repr(e)[-1500:], flush=True)
    torch.cuda.synchronize(dev)
    print('MR_DONE', _okh_clock.perf_counter()-start, 'no_new_timed_path=True', flush=True)
    return state
