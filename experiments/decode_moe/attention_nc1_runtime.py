"""Appended standalone Attention setup; no real input values enter calibration."""
_NC1_STATES = {}
_nc1_parent_kernel_fn = kernel_fn


class _NC1Direct:
    def __init__(self, module, expected):
        self.module = module
        self.expected = expected

    def run(self, *args):
        if _nc1_run_family(args) != self.expected:
            raise RuntimeError('nc1 diagnostic did not select its screened entry')
        return self.module.run_nc1(*args)


class _NC1Dispatch:
    def __init__(self, parent, candidate=None, enabled=()):
        self.parent = parent
        self.candidate = candidate
        self.enabled = frozenset(enabled)

    def run(self, *args):
        if self.candidate is not None and _nc1_run_family(args) in self.enabled:
            return self.candidate.run_nc1(*args)
        return self.parent.run(*args)


def _nc1_resources_ok(values):
    if len(values) != 12:
        return False
    for i, expected in enumerate(((67584, 256), (58368, 128))):
        regs, local, static, shared, threads, blocks = values[6*i:6*i+6]
        if not (0 < regs <= 255 and local == static == 0 and
                (shared, threads) == expected and blocks >= 1):
            return False
    return True


def _nc1_reset_workspace(dev):
    # Only the parent's internal scratch is reset; caller-owned KV caches are
    # never passed here. This also isolates a failed diagnostic from fallback.
    for key, buffers in _WS.items():
        if key[0] == str(dev):
            for tensor in buffers:
                tensor.zero_()


def _nc1_data(B, ctx, sliding, dev, seed, page=1, pad=0, hkv=2, scale=1., coherent=False):
    # Independent synthetic inputs only. Shapes/config come from this fixed
    # diagnostic packet, not from a caller's activation or sequence tensors.
    gen = torch.Generator(device=dev).manual_seed(seed)
    max_pages = (ctx + 1 + page - 1) // page + pad
    pages = B * max_pages
    table = torch.randperm(pages, generator=gen, device=dev).to(torch.int32).reshape(B, max_pages)
    shape = (pages, page, hkv, 64)
    def cache(normalize):
        values = torch.randn(shape, device=dev, generator=gen)
        if coherent:
            base = torch.randn((1, 1, hkv, 64), device=dev, generator=gen)
            values.mul_(.2).add_(base)
        if normalize:
            values.mul_(torch.rsqrt(values.square().mean(-1, keepdim=True) + 1e-6))
        return (values / scale).clamp_(-448., 448.).to(torch.float8_e4m3fn)
    data = dict(qkv=torch.empty((B, 1, (18*hkv)*64), dtype=torch.bfloat16, device=dev),
                seq_lens=torch.empty(B, dtype=torch.int32, device=dev),
                cache_loc=torch.empty(B, dtype=torch.int64, device=dev),
                k_cache=cache(True), v_cache=cache(False), page_table=table,
                k_scale=torch.tensor(scale, dtype=torch.float32, device=dev),
                v_scale=torch.tensor(scale, dtype=torch.float32, device=dev),
                scheduler_metadata=None,
                config=dict(batch=B, q_len=1, num_q_heads=8*hkv, num_kv_heads=hkv,
                            head_dim=64, page_size=page, window=2048 if sliding else 0,
                            rope_theta=10000. if sliding else 0.,
                            qk_scale_factor=5.4730064863875, sm_scale=.125, kv_scale=scale))
    _nc1_mutate(data, ctx, seed, 0)
    return data


def _nc1_mutate(data, ctx, seed, variant):
    dev = data['qkv'].device
    gen = torch.Generator(device=dev).manual_seed(seed + 101*variant)
    data['qkv'].normal_(generator=gen)
    if variant == 1:
        data['qkv'].zero_()
    elif variant == 2:
        data['qkv'].mul_(8.)
    B, P = data['config']['batch'], data['config']['page_size']
    step = min(251, max(1, ctx // (2*B)))
    lengths = torch.tensor([max(0, ctx - (b % 5)*step - variant) for b in range(B)],
                           dtype=torch.int32, device=dev)
    data['seq_lens'].copy_(lengths)
    logical = lengths.to(torch.int64)
    physical = torch.gather(data['page_table'].long(), 1, (logical // P)[:, None]).flatten()
    data['cache_loc'].copy_(physical * P + logical % P)


def _nc1_byte_probe(data, ctx, seed, parent, candidate, expected):
    arms = {'parent': parent, 'parent_repeat': parent, 'candidate': _NC1Direct(candidate, expected)}
    captured = {}
    for name, module in arms.items():
        _launch(data, _mod=module)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = _launch(data, _mod=module)
        captured[name] = (graph, out)
    for variant in range(3):
        _nc1_mutate(data, ctx, seed, variant)
        ref = None
        for name, (graph, out) in captured.items():
            graph.replay()
            torch.cuda.synchronize()
            if not bool(torch.isfinite(out).all()):
                print('NC1_FINITE_FAIL', expected, name, variant, flush=True)
                return False
            snap = tuple(t.contiguous().view(torch.uint8).clone()
                         for t in (out, data['k_cache'], data['v_cache']))
            if name == 'parent':
                ref = snap
            elif not _okh_same(ref, snap):
                mismatch = [int((a != b).sum()) for a, b in zip(ref, snap)]
                print('NC1_BYTES_FAIL', expected, name, variant, mismatch, flush=True)
                return False
    print('NC1_BYTES_PASS', expected, data['config']['batch'], ctx,
          data['config']['page_size'], 'three_changing_inputs;output_and_entire_cache_bytes', flush=True)
    return True


def _nc1_initialize(dev):
    start = _okh_clock.perf_counter()
    built = {}
    def build():
        try:
            with torch.cuda.device(dev):
                built['candidate'] = _nc1_build_native()
        except Exception as exc:
            built['error'] = repr(exc)[-1500:]
    thread = _okh_bg(build)
    # Both compilation paths complete before any diagnostic graph capture.
    try:
        with torch.cuda.device(dev):
            parent = _module()
    finally:
        thread.join()
    enabled = []
    candidate = built.get('candidate')
    try:
        if candidate is None:
            raise RuntimeError('nc1 build unavailable: ' + built.get('error', 'unknown'))
        values = list(candidate.nc1_resources())
        print('NC1_RESOURCES', values, flush=True)
        if not _nc1_resources_ok(values):
            raise RuntimeError('nc1 resource guard rejected before probes')
        # Six correctness configurations per family, including the performance
        # cohorts. Each configuration replays three changing inputs/lengths.
        groups = {
            'spec_one_L1': [(1,10,True,1,0,2,1.,False),
                            (1,2047,True,1,17,2,1./64,False),
                            (1,8192,True,1,9,4,1.,True),
                            (1,63,False,1,0,4,1.,False),
                            (1,80000,True,1,0,2,1.,False),
                            (1,80000,True,1,17,2,1.,True)],
            'flat_L1': [(2,129,False,64,0,2,1./64,False),
                        (3,4096,False,1,31,2,1.,True),
                        (4,8192,False,64,0,4,1.,False),
                        (5,16384,False,1,7,2,1.,True),
                        (16,80000,False,1,0,2,1.,False),
                        (32,80000,False,1,17,2,1.,True)]}
        for family, cases in groups.items():
            valid = True
            for i, (B, ctx, sw, page, pad, hkv, scale, coherent) in enumerate(cases):
                if not _okh_budget(start, 274.):
                    valid = False
                    print('NC1_BUDGET', family, 'before_correctness', flush=True)
                    break
                data = _nc1_data(B, ctx, sw, dev, 2026103100+i, page, pad, hkv, scale, coherent)
                valid = _nc1_byte_probe(data, ctx, 2026103100+i, parent, candidate, family)
                del data
                if not valid:
                    break
            if not valid:
                continue
            # Fixed cohorts and distributions, two arm orders. No parameter sweep.
            targets = ((1,80000,True),(1,8192,True)) if family == 'spec_one_L1' else (
                       (16,80000,False),(32,80000,False))
            choices = []
            for B, ctx, sw in targets:
                for coherent in (False, True):
                    if not _okh_budget(start, 278.):
                        break
                    data = _nc1_data(B, ctx, sw, dev, 2026103177, coherent=coherent)
                    direct = _NC1Direct(candidate, family)
                    fns = {'parent': lambda: _launch(data, _mod=parent),
                           'candidate': lambda: _launch(data, _mod=direct)}
                    for order in (('parent','candidate'), ('candidate','parent')):
                        if not _okh_budget(start, 278.):
                            break
                        times = _okh_time({name:fns[name] for name in order}, dev, reps=25)
                        choices.append(_okh_pick(times, 'parent', .97) == 'candidate')
                        print('NC1_TIMES', family, B, ctx, coherent, order, times, flush=True)
                    del data
            nearby = (1,2047,True,1,17) if family == 'spec_one_L1' else (3,4096,False,1,31)
            nearby_ok = []
            if len(choices) == 8 and all(choices):
                B, ctx, sw, page, pad = nearby
                if _okh_budget(start, 278.):
                    data = _nc1_data(B, ctx, sw, dev, 2026103199, page=page, pad=pad)
                    direct = _NC1Direct(candidate, family)
                    fns = {'parent': lambda: _launch(data, _mod=parent),
                           'candidate': lambda: _launch(data, _mod=direct)}
                    for order in (('parent','candidate'), ('candidate','parent')):
                        if not _okh_budget(start, 278.):
                            break
                        times = _okh_time({name:fns[name] for name in order}, dev, reps=25)
                        nearby_ok.append(times['candidate'] <= 1.02 * times['parent'])
                        print('NC1_NEARBY', family, order, times, flush=True)
                    del data
            if len(choices) == 8 and all(choices) and len(nearby_ok) == 2 and all(nearby_ok):
                enabled.append(family)
            print('NC1_SELECT', family, family in enabled, flush=True)
    except Exception as exc:
        enabled = []
        print('NC1_ERROR', repr(exc)[-1500:], flush=True)
    torch.cuda.synchronize()
    _nc1_reset_workspace(dev)
    torch.cuda.synchronize()
    print('NC1_DONE', _okh_clock.perf_counter() - start, enabled, flush=True)
    return _NC1Dispatch(parent, candidate, enabled)


def kernel_fn(data):
    # Original support/index guards remain ahead of all private dispatch.
    if not _supported(data['config']) or not _fits(data) or data['config']['q_len'] != 1:
        return _nc1_parent_kernel_fn(data)
    dev = data['qkv'].device
    state = _NC1_STATES.get(str(dev))
    if state is None:
        if torch.cuda.is_current_stream_capturing():
            return _nc1_parent_kernel_fn(data)
        with torch.cuda.device(dev):
            state = _nc1_initialize(dev)
        _NC1_STATES[str(dev)] = state
    return _launch(data, _mod=state)
