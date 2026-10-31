"""Joint pooler/router diagnostic during Dense preparation; public Dense calls retain parent."""


def _rs_find(d, key, depth=0):
    if not isinstance(d, dict) or depth > 6:
        return None
    if key in d:
        return d[key]
    for value in d.values():
        found = _rs_find(value, key, depth + 1)
        if found is not None:
            return found
    return None


def _rs_run(ext, plan, xb, xr, state):
    m = xb.shape[0]
    out = torch.empty((m, 4096), dtype=torch.bfloat16, device=xb.device)
    ow = torch.empty((m, 8), dtype=torch.float32, device=xb.device)
    oi = torch.empty((m, 8), dtype=torch.int32, device=xb.device)
    wr, wk, ku, split = plan['cfg']
    rc = ext.r2_launch(plan['wp'].data_ptr(), xb.data_ptr(), out.data_ptr(), plan['esc'].data_ptr(),
                       m, 4096, 8192, wr, wk, ku, split, plan['t01'], plan['u01'],
                       torch.cuda.current_stream().cuda_stream, xr.data_ptr(), state['weight'].data_ptr(),
                       state['bias'].data_ptr(), state['parts'].data_ptr(), state['counter'].data_ptr(),
                       ow.data_ptr(), oi.data_ptr(), 0)
    if rc:
        raise RuntimeError('joint dense unit launch failed:' + str(rc))
    return out, ow, oi


def _rs_probe(modules, plan, states, dev):
    for m in (1, 8, 2, 4):
        xb = torch.zeros((m, 8192), dtype=torch.bfloat16, device=dev)
        xr = torch.zeros((m, 8192), dtype=torch.float32, device=dev)
        graphs, outputs = {}, {}
        for arm in ('parent', 'parent_repeat', 'candidate'):
            ext = modules['candidate' if arm == 'candidate' else 'parent']
            _rs_run(ext, plan, xb, xr, states[arm])
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                outputs[arm] = _rs_run(ext, plan, xb, xr, states[arm])
            graphs[arm] = graph
        for variant in range(3):
            g = torch.Generator(device=dev).manual_seed(2026103105 + 31 * m + variant)
            fresh = torch.randn((m, 8192), device=dev, dtype=torch.float32, generator=g)
            if variant == 0:
                fresh.zero_()
            elif variant == 2:
                fresh[:, -1] = 128.03125
            xr.copy_(fresh)
            xb.copy_(fresh.to(torch.bfloat16))
            for arm in states:
                states[arm]['bias'].zero_()
            snapshots = {}
            for arm in ('parent', 'parent_repeat', 'candidate'):
                graphs[arm].replay()
                torch.cuda.synchronize()
                values = outputs[arm] + (states[arm]['parts'], states[arm]['counter'])
                snapshots[arm] = tuple(t.contiguous().view(torch.uint8).clone() for t in values)
                if arm != 'parent' and not _okh_same(snapshots['parent'], snapshots[arm]):
                    print('RS_BYTES_FAIL', m, variant, arm,
                          [int((a != b).sum()) for a, b in zip(snapshots['parent'], snapshots[arm])], flush=True)
                    return False
                if int(states[arm]['counter'].item()) != 0:
                    raise RuntimeError('joint dense unit left nonzero arrivals')
            print('RS_BYTES_PASS', m, variant, 'pooler/router_weights/router_ids/full_parts/counter', flush=True)
        del graphs, outputs
    return True


def _rs_diagnose(state, weights, modules, start):
    plan = (_rs_find(state, 'v18') or {}).get('plans', {}).get(('pooler', 1))
    if plan is None or plan.get('fmt') != 1 or tuple(plan.get('cfg', ())) != (2, 16, 8, 1):
        print('RS_UNAVAILABLE exact retained pooler plan', flush=True)
        return {'accepted_unit': False, 'reason': 'no matching plan'}
    rw = weights.get('router')
    if rw is None or rw.dtype != torch.float32 or tuple(rw.shape) != (256, 8192) or not rw.is_contiguous():
        print('RS_UNAVAILABLE original FP32 router weights', flush=True)
        return {'accepted_unit': False, 'reason': 'router contract'}
    dev = rw.device
    states = {arm: {'weight': rw, 'bias': torch.zeros(256, dtype=torch.float32, device=dev),
                    'parts': torch.zeros((8, 8, 256), dtype=torch.float32, device=dev),
                    'counter': torch.zeros(1, dtype=torch.int32, device=dev)}
              for arm in ('parent', 'parent_repeat', 'candidate')}
    if not _okh_budget(start, 268.) or not _rs_probe(modules, plan, states, dev):
        return {'accepted_unit': False, 'reason': 'budget or exact-byte gate'}
    # Confirm the real private kernel; this is part of the same diagnostic job.
    if not _okh_budget(start, 274.):
        return {'accepted_unit': False, 'reason': 'marker budget'}
    marker_xr = torch.zeros((1, 8192), dtype=torch.float32, device=dev)
    marker_xb = marker_xr.to(torch.bfloat16)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as profile:
        _rs_run(modules['candidate'], plan, marker_xb, marker_xr, states['candidate'])
        torch.cuda.synchronize()
    markers = sorted({event.name for event in profile.events() if 'sidecar_gemv' in event.name})
    print('RS_MARKERS', markers, flush=True)
    if not markers:
        return {'accepted_unit': False, 'reason': 'missing actual sidecar kernel'}
    results = []
    for population in ('normal', 'spike'):
        g = torch.Generator(device=dev).manual_seed(2026103119)
        xr = torch.randn((1, 8192), dtype=torch.float32, device=dev, generator=g)
        if population == 'spike':
            xr[:, -1] = 128.03125
        xb = xr.to(torch.bfloat16)
        for arm in states:
            states[arm]['bias'].copy_(torch.linspace(-.01, .01, 256, dtype=torch.float32, device=dev))
        fns = {arm: (lambda arm=arm: _rs_run(modules['parent' if arm == 'parent_repeat' else arm], plan, xb, xr, states[arm]))
               for arm in ('parent', 'parent_repeat', 'candidate')}
        for order in (('parent', 'candidate'), ('candidate', 'parent')):
            if not _okh_budget(start, 278.):
                return {'accepted_unit': False, 'reason': 'timing budget', 'comparisons': results}
            samples = {}
            med = _okh_time({arm: fns[arm] for arm in order}, dev, reps=25, samples=samples)
            result = {'population': population, 'order': order, 'median_us': med, 'raw_us': samples,
                      'pass': med['candidate'] < .97 * med['parent']}
            results.append(result)
            print('RS_TIMES', result, flush=True)
    accepted = len(results) == 4 and all(x['pass'] for x in results)
    controls = []
    if accepted:
        for order in (('parent', 'parent_repeat'), ('parent_repeat', 'parent')):
            if not _okh_budget(start, 278.):
                return {'accepted_unit': False, 'reason': 'control budget', 'comparisons': results}
            samples = {}
            med = _okh_time({arm: fns[arm] for arm in order}, dev, reps=25, samples=samples)
            control = {'order': order, 'median_us': med, 'raw_us': samples,
                       'pass': abs(med['parent_repeat'] / med['parent'] - 1.) <= .02}
            controls.append(control)
            print('RS_CONTROL', control, flush=True)
        accepted = len(controls) == 2 and all(x['pass'] for x in controls)
    print('RS_UNIT_ACCEPT', accepted, 'diagnostic only;public Dense and Decode unchanged', flush=True)
    return {'accepted_unit': accepted, 'comparisons': results, 'unchanged_controls': controls}


def prepare(weights):
    start = _okh_clock.perf_counter()
    # Build before any graph capture. No live background compiler enters timed calls.
    from torch.utils.cpp_extension import load_inline as _rs_load_inline
    import hashlib as _rs_hashlib
    modules = {}
    build_error = None
    try:
        for arm, source in (('parent', RS_PARENT_CUDA), ('candidate', RS_CANDIDATE_CUDA)):
            tag = _rs_hashlib.sha256(source.encode()).hexdigest()[:16]
            module = _rs_load_inline('okc_rs_joint_' + arm + '_' + tag, RS_CPP, cuda_sources=source,
                                     functions=['r2_launch', 'r2_resources'], extra_cuda_cflags=['-O3', '-lineinfo'],
                                     no_implicit_headers=True, with_pytorch_error_handling=False, verbose=False)
            resources = module.r2_resources()
            print('RS_RESOURCES', arm, resources, flush=True)
            if len(resources) != 6 or resources[:4] != [64, 0, 32768, 16384] or resources[4] != 1 or resources[5] < 2:
                raise RuntimeError('joint dense unit resource guard:' + repr(resources))
            modules[arm] = module
    except Exception as exc:
        build_error = repr(exc)[-1400:]
        print('RS_BUILD_ERROR', build_error, flush=True)
    state = _rs_parent_prepare(weights)
    try:
        result = _rs_diagnose(state, weights, modules, start) if len(modules) == 2 else {'accepted_unit': False, 'reason': build_error}
    except Exception as exc:
        result = {'accepted_unit': False, 'reason': repr(exc)[-1400:]}
        print('RS_ERROR', result['reason'], flush=True)
    state['rs_joint_diagnostic'] = result
    print('RS_DONE', _okh_clock.perf_counter() - start, result['accepted_unit'], flush=True)
    return state


def kernel_fn(data, state):
    return _rs_parent_kernel_fn(data, state)
