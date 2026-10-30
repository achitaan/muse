"""Prepare-only gates for the new routing prologue, with exact fallbacks."""

_tr_parent_prepare = prepare
_tr_parent_kernel_fn = kernel_fn


def _tr_prepare_build(sfx):
    mods = {}
    for name, defs in [('tiny', TINY_DEFINES), ('tiny2', TINY2_DEFINES)]:
        mods[name] = _ext(tuple(defs) + tuple(sfx) + ('-DOKC_ROUTE_WARP=1',))
    return mods


def _tr_probe_data(m, state, dev):
    return dict(config=dict(local_start=state['lo'], local_count=state['cnt']),
                hidden_states=torch.empty((m, HIDDEN), dtype=torch.bfloat16, device=dev),
                topk_ids=torch.empty((m, 8), dtype=torch.int32, device=dev),
                topk_weights=torch.empty((m, 8), dtype=torch.float32, device=dev))


def _tr_mutate(data, state, pattern, seed):
    # Synthetic routing is development evidence only, never inferred from a
    # timed input. Each row keeps eight distinct global expert IDs.
    import random
    rng = random.Random(20261030 + 1009 * seed + 31 * data['hidden_states'].shape[0])
    lo, cnt = state['lo'], state['cnt']
    local = list(range(lo, lo + cnt))
    remote = [e for e in range(256) if not lo <= e < lo + cnt]
    hot = rng.sample(local, 8)
    rows = []
    for t in range(data['hidden_states'].shape[0]):
        n = {'none': 0, 'one': 1, 'four': 4, 'all': 8}.get(pattern, 1 + t % 3)
        rows.append(rng.sample(hot, n) + rng.sample(remote, 8 - n))
    data['topk_ids'].copy_(torch.tensor(rows, dtype=torch.int32, device=data['topk_ids'].device))
    gen = torch.Generator(device=data['hidden_states'].device).manual_seed(151 + seed)
    data['hidden_states'].normal_(generator=gen)
    if pattern == 'none':
        data['hidden_states'].fill_(float('nan'))  # no invalid-route arithmetic
    data['topk_weights'].uniform_(.1, 1., generator=gen)
    data['topk_weights'].div_(data['topk_weights'].sum(-1, keepdim=True))


def _tr_probe(data, parent_state, candidate_state):
    _tr_mutate(data, parent_state, 'one', 0)
    graphs = []
    for st in (parent_state, candidate_state):
        _tr_launch(data, st)
        torch.cuda.synchronize()
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            out = _tr_launch(data, st)
        graphs.append((gr, out))
    for pattern in ('none', 'one', 'four', 'all', 'skew'):
        for seed in range(3):
            _tr_mutate(data, parent_state, pattern, seed)
            snaps = []
            for gr, out in graphs:
                gr.replay()
                torch.cuda.synchronize()
                snaps.append(out.clone())
            if (not all(bool(torch.isfinite(v).all()) for v in snaps)
                    or not torch.equal(snaps[0].view(torch.uint8), snaps[1].view(torch.uint8))):
                print('TR8_BITWISE_FAIL', data['hidden_states'].shape[0], pattern, seed, flush=True)
                return False
            if pattern == 'none' and bool((snaps[1] != 0).any()):
                return False
    return True


def prepare(weights):
    start = _okh_clock.perf_counter()
    dev = weights['w13'].device
    lo, cnt = LOCAL_START, LOCAL_EXPERTS
    prmt = PRMT_SCALES and _scales_prmt_exact(weights['w13_sf'][lo:lo + cnt], weights['w2_sf'][lo:lo + cnt])
    sfx = ('-DOKC_PRMTSC=1',) if prmt else ()
    built = {}
    def build():
        try:
            with torch.cuda.device(dev):
                built['mods'] = _tr_prepare_build(sfx)
        except Exception as e:
            built['error'] = repr(e)[-1500:]
    thread = _okh_bg(build)
    state = _tr_parent_prepare(weights)
    state['tr8'] = {}
    # No compiler thread survives prepare or runs alongside graph capture.
    thread.join()
    if 'mods' not in built or not _okh_budget(start, 278.):
        print('TR8_SKIP', built.get('error'), _okh_clock.perf_counter() - start, flush=True)
        return state
    try:
        assert tuple(state.get('sfx', ())) == sfx
        mods = built['mods']
        for m in range(1, 9):
            if not _okh_budget(start, 278.):
                break
            ext = mods['tiny' if m <= 4 else 'tiny2']
            probe_state = dict(state, tr8={m: ext})
            data = _tr_probe_data(m, state, dev)
            ok = _tr_probe(data, state, probe_state)
            print('TR8_BITWISE', m, ok, 'five_route_patterns_three_inputs', flush=True)
            if not ok:
                continue
            passes = []
            for pattern in ('one', 'skew'):
                _tr_mutate(data, state, pattern, 7)
                fns = {'parent': lambda: _tr_launch(data, state),
                       'candidate': lambda: _tr_launch(data, probe_state)}
                for order in [('parent', 'candidate'), ('candidate', 'parent')]:
                    if not _okh_budget(start, 278.):
                        break
                    med = _okh_time({k: fns[k] for k in order}, dev, reps=25)
                    pick = _okh_pick(med, 'parent', .97)
                    passes.append(pick == 'candidate')
                    print('TR8_TIME', m, pattern, ','.join(order), med, 'pick=', pick, flush=True)
            if len(passes) == 4 and all(passes):
                state['tr8'][m] = ext
            print('TR8_SELECT', m, m in state['tr8'], flush=True)
    except Exception as e:
        state['tr8'] = {}
        print('TR8_ERROR', repr(e)[-1500:], flush=True)
    torch.cuda.synchronize()
    print('TR8_DONE', _okh_clock.perf_counter() - start, sorted(state['tr8']), flush=True)
    return state


def kernel_fn(data, state):
    if data['hidden_states'].shape[0] in state.get('tr8', {}):
        return _tr_launch(data, state)
    return _tr_parent_kernel_fn(data, state)
