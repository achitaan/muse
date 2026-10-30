"""Prepare-only causal norm/R2f diagnostic; timed Dense kernel is untouched."""

_NP_R_EXT = None
_NP_R_ERROR = None
_NP_PUBLIC = None
_NP_PUBLIC_ERROR = None
_PDLS_EXT = None
_PDLS_ERROR = None
_PDLI_EXT = {}


def _np_build_r():
    global _NP_R_EXT, _NP_R_ERROR
    try:
        assert hashlib.sha256(_NP_R_CUDA.encode()).hexdigest() == _NP_R_SHA
        _NP_R_EXT = load_inline('okc_np_r_' + _NP_R_SHA[:12], _NP_R_CPP,
            cuda_sources=_NP_R_CUDA, functions=_NP_R_FUNCTIONS,
            extra_cuda_cflags=['-O3', '-lineinfo', '-gencode=arch=compute_90a,code=sm_90a'],
            no_implicit_headers=True, with_pytorch_error_handling=False, verbose=False)
        for m in (1, 8):
            _PDLI_EXT['r' + str(m)] = _PdliFns(_NP_R_EXT, 'r' + str(m))
        print('NP_R_BUILD', _NP_R_SHA, flush=True)
    except Exception as e:
        _NP_R_ERROR = repr(e)[-1500:]
        print('NP_R_BUILD_ERROR', _NP_R_ERROR, flush=True)


def _np_build_public():
    global _NP_PUBLIC, _NP_PUBLIC_ERROR
    try:
        from pathlib import Path
        from sglang.kernels.ops.muse_spark_v12 import norm_residual as module
        digest = hashlib.sha256(Path(module.__file__).read_text().encode()).hexdigest()
        if digest != _NP_PUBLIC_SHA:
            raise RuntimeError('NP public adapter identity: ' + digest)
        # Compile without launching a kernel; the public adapter uses this cache.
        module._jit_module(8192, 256, True, 1)
        _NP_PUBLIC = module
        print('NP_PUBLIC_BUILD', digest, flush=True)
    except Exception as e:
        _NP_PUBLIC_ERROR = repr(e)[-1500:]
        print('NP_PUBLIC_BUILD_ERROR', _NP_PUBLIC_ERROR, flush=True)


_np_r_thread = _okh_bg(_np_build_r)
_np_public_thread = _okh_bg(_np_build_public)
_pdls_thread = _okh_bg(_pdls_build)
_np_parent_prepare = prepare


def _np_find(d, key, depth=0):
    if not isinstance(d, dict) or depth > 6:
        return None
    if key in d:
        return d[key]
    for value in d.values():
        found = _np_find(value, key, depth + 1)
        if found is not None:
            return found
    return None


def _r2_pool_route(*args):
    raise RuntimeError('NP unexpected fallback in private diagnostic route')


def _np_original(stream, branch, p):
    return _NP_PUBLIC.fused_norm_residual(branch, stream, p['post_eps'], p['pre_eps'],
        pre_weight=p['pre'], alpha=p['alpha'], beta=p['beta'],
        want_router=True, pre_norm_bf16=False)


def _np_probe(diag, m, gen, dev, start, report):
    branch = torch.randn((m, 8192), device=dev, generator=gen).to(torch.bfloat16)
    stream = torch.randn((m, 8192), device=dev, generator=gen)
    for scalar_gate in (False, True):
        if not _okh_budget(start, 274.):
            report['budget_stop'] = True
            return
        p = dict(ready=True, pre=(1. + .05 * torch.randn((8192,), device=dev, generator=gen)).to(torch.bfloat16),
            alpha=.1 * torch.randn((1 if scalar_gate else 8192,), device=dev, generator=gen),
            beta=.1 * torch.randn((1 if scalar_gate else 8192,), device=dev, generator=gen),
            post_eps=1e-8, pre_eps=1e-5)
        local = dict(diag, pdls=p)
        def mutate(seed):
            branch.normal_(generator=gen); stream.normal_(generator=gen)
            if seed == 1:
                branch.zero_(); stream.zero_()
            elif seed == 2:
                branch.mul_(16.); stream.mul_(.001)
        def norm(on):
            return (_pdls_boundary(stream, branch, local) if on else _np_original(stream, branch, p))
        label = 'NP_norm_M' + str(m) + ('_scalar' if scalar_gate else '_channel')
        same = _pdli_probe({'parent': lambda: norm(False), 'early': lambda: norm(True)}, mutate, label=label)
        report['norm_bytes'].append(dict(m=m, scalar_gate=scalar_gate, equal=bool(same)))
        print('NP_BYTES', label, same, flush=True)
        if scalar_gate or not same or 'r' + str(m) not in diag['pdli']['plans']:
            continue
        def chain(on):
            b = norm(on)
            return tuple(b) + tuple(_pdli_route(b[0], b[2], local))
        label = 'NP_pair_M' + str(m)
        same = _pdli_probe({'parent': lambda: chain(False), 'early': lambda: chain(True)}, mutate, label=label)
        item = dict(m=m, equal=bool(same), orders=[])
        report['pairs'].append(item)
        print('NP_BYTES', label, same, flush=True)
        if not same:
            continue
        # Restore ordinary inputs after the adversarial byte probes.
        mutate(0)
        for order in (('parent', 'early'), ('early', 'parent')):
            if not _okh_budget(start, 278.):
                report['budget_stop'] = True
                break
            arms = {'parent': lambda: chain(False), 'early': lambda: chain(True)}
            times = _okh_time({key: arms[key] for key in order}, dev, reps=25)
            saving = times['parent'] - times['early']
            ratio = times['early'] / times['parent']
            passed = _okh_pick(times, 'parent', .97) == 'early' and saving >= 1.5
            item['orders'].append(dict(order=list(order), parent_us=times['parent'], early_us=times['early'],
                saving_us=saving, ratio=ratio, passes=passed))
            print('NP_TIME', m, order, times, 'saving_us=', saving, 'passes=', passed, flush=True)
        item['causal_gate'] = len(item['orders']) == 2 and all(x['passes'] for x in item['orders'])


def prepare(weights):
    start = _okh_clock.perf_counter()
    state = _np_parent_prepare(weights)
    for thread in (_np_r_thread, _np_public_thread, _pdls_thread):
        thread.join()
    report = dict(timed_path='unchanged Dense parent', enables_nothing=True,
        norm_bytes=[], pairs=[], budget_stop=False, errors=[], resources={})
    try:
        if _NP_R_EXT is None or _NP_PUBLIC is None or _PDLS_EXT is None:
            raise RuntimeError(str((_NP_R_ERROR, _NP_PUBLIC_ERROR, _PDLS_ERROR)))
        dev = weights['router'].device
        prop = torch.cuda.get_device_properties(dev)
        if prop.major != 9 or prop.multi_processor_count < 129:
            raise RuntimeError('NP unsupported device/resources: ' + prop.name)
        resource = [int(_PDLS_EXT.query(i)) for i in range(4)]
        report['resources']['norm'] = resource
        if resource[0] <= 0 or resource[1] != 0 or resource[3] <= 0:
            raise RuntimeError('NP norm resources: ' + str(resource))
        gen = torch.Generator(device=dev).manual_seed(202610301)
        plans = {}
        for m, family in ((1, 'v18'), (8, 'v10_sm2')):
            pp = (_np_find(state, family) or {}).get('plans', {}).get(('pooler', m))
            if pp is None or pp.get('fmt') != 1 or tuple(pp.get('cfg', ())) != (2,16,8,1):
                report['errors'].append('NP missing exact packed pooler plan M' + str(m))
                continue
            info = _PDLI_EXT['r' + str(m)].pdli_info
            regs, blocks = int(info(m, 2)), int(info(m, 1))
            report['resources']['r' + str(m)] = [regs, blocks]
            if regs < 0 or regs % 1000 != 0 or blocks < 1:
                continue
            plans['r' + str(m)] = pp
        diag = dict(pdli=dict(plans=plans, routing_scratch={
                'parts':torch.empty((8,8,256), dtype=torch.float32, device=dev),
                'counter':torch.zeros((1,), dtype=torch.int32, device=dev)}),
            rf=dict(weight=weights['router'].contiguous(),
                    bias=.05 * torch.randn((256,), device=dev, generator=gen)))
        print('NP_RESOURCES', report['resources'], flush=True)
        for m in (1, 8):
            if not _okh_budget(start, 274.):
                report['budget_stop'] = True
                break
            _np_probe(diag, m, gen, dev, start, report)
        torch.cuda.synchronize()
    except Exception as e:
        report['errors'].append(repr(e)[-2000:])
        print('NP_ERROR', report['errors'][-1], flush=True)
    report['prepare_s'] = _okh_clock.perf_counter() - start
    # Compact terminal report survives the official 16k-character log tail.
    import base64, zlib, json
    payload = base64.b64encode(zlib.compress(json.dumps(report, allow_nan=False).encode(), 9)).decode()
    assert len(payload) < 6000
    print('NP_SUMMARY_Z64', payload, flush=True)
    print('NP_DONE', report['prepare_s'], 'enables_nothing=True', flush=True)
    return state
