"""Prepare-only fixed O/U ceilings and real norm/FPR chains at M16/M32."""

_P16_EXT = None
_D16_BUILD_ERROR = None
_PDLN_EXT = None
_PDLN_ERROR = None
_D16_PUBLIC = None
_D16_PUBLIC_ERROR = None
_D16_FPR_ERROR = None


def _d16_build_bundle():
    global _P16_EXT, _D16_BUILD_ERROR
    try:
        assert hashlib.sha256(_D16_CUDA.encode()).hexdigest() == _D16_CUDA_SHA
        _P16_EXT = load_inline('okc_d16_' + _D16_CUDA_SHA[:12], _D16_CPP,
            cuda_sources=_D16_CUDA, functions=_D16_FUNCTIONS,
            extra_cuda_cflags=['-O3', '-lineinfo', '-gencode=arch=compute_90a,code=sm_90a'],
            no_implicit_headers=True, with_pytorch_error_handling=False, verbose=False)
        print('D16_BUILD', _D16_CUDA_SHA, flush=True)
    except Exception as e:
        _D16_BUILD_ERROR = repr(e)[-1600:]
        print('D16_BUILD_ERROR', _D16_BUILD_ERROR, flush=True)


def _d16_build_public():
    global _D16_PUBLIC, _D16_PUBLIC_ERROR
    try:
        from pathlib import Path
        from sglang.kernels.ops.muse_spark_v12 import norm_residual as module
        digest = hashlib.sha256(Path(module.__file__).read_text().encode()).hexdigest()
        if digest != _D16_PUBLIC_SHA:
            raise RuntimeError('D16 public adapter identity: ' + digest)
        module._jit_module(8192, 256, True, 1)
        _D16_PUBLIC = module
        print('D16_PUBLIC_BUILD', digest, flush=True)
    except Exception as e:
        _D16_PUBLIC_ERROR = repr(e)[-1600:]
        print('D16_PUBLIC_BUILD_ERROR', _D16_PUBLIC_ERROR, flush=True)


def _d16_build_fpr():
    global _D16_FPR_ERROR
    try:
        _fpr_get_ext()
        print('D16_FPR_BUILD', _D16_FPR_ORIGINAL_SHA, flush=True)
    except Exception as e:
        _D16_FPR_ERROR = repr(e)[-1600:]
        print('D16_FPR_BUILD_ERROR', _D16_FPR_ERROR, flush=True)


_d16_threads = [_okh_bg(_d16_build_bundle), _okh_bg(_d16_build_public),
                _okh_bg(_pdln_build), _okh_bg(_d16_build_fpr)]
_d16_parent_prepare = prepare
_d16_parent_kernel_fn = kernel_fn


def _r2_find(d, key, depth=0):
    if not isinstance(d, dict) or depth > 6:
        return None
    if key in d:
        return d[key]
    for value in d.values():
        found = _r2_find(value, key, depth + 1)
        if found is not None:
            return found
    return None


def _achita_dense_project(*args):
    raise RuntimeError('D16 unexpected private fallback; no new-path evidence')


def _d16_reference(x, layer, diag):
    n = 8192
    return _d16_parent_kernel_fn(dict(x=x,
        config=dict(m=x.shape[0], n=n, k=x.shape[1], layer=layer)), diag['achita_dense'])


def _d16_spin(src, dst):
    dst.copy_(src)
    rc = _P16_EXT.d16_spin_start(src.data_ptr(), dst.data_ptr(), src.shape[0], src.shape[1],
                               torch.cuda.current_stream().cuda_stream, 5000)
    if rc:
        raise RuntimeError('D16 spin launch: ' + str(rc))


def _d16_times(label, arms, dev, start, row):
    row['orders'] = []
    order0 = tuple(arms)
    for order in (order0, tuple(reversed(order0))):
        if not _okh_budget(start, 278.):
            row['budget_stop'] = True
            break
        times = _okh_time({key: arms[key] for key in order}, dev, reps=25)
        saving = times['parent'] - times['early']
        passed = times['early'] < .97 * times['parent'] and saving >= 1.5
        item = dict(order=list(order), times_us=times, saving_us=saving,
                    ratio=times['early'] / times['parent'], passes=passed)
        if 'late' in times:
            item['early_signal_vs_late_saving_us'] = times['late'] - times['early']
            item['early_signal_vs_late_ratio'] = times['early'] / times['late']
            item['causal_signal_pass'] = (times['early'] < .97 * times['late']
                                          and item['early_signal_vs_late_saving_us'] >= 1.5)
            item['passes'] = passed and item['causal_signal_pass']
        row['orders'].append(item)
        print('D16_TIME', label, item, flush=True)
    row['handoff_gate'] = len(row['orders']) == 2 and all(x['passes'] for x in row['orders'])


def _d16_projection_probe(diag, m, layer, gen, dev, start, report):
    if not _okh_budget(start, 274.):
        report['budget_stop'] = True
        return
    key = m if layer == 'o' else 'u' + str(m)
    if key not in diag['p16']['plans']:
        report['skipped'].append('No exact eligible plan: ' + layer + str(m))
        return
    k = 1024 if layer == 'o' else 4096
    if (m, layer, 8192, k) in diag['achita_dense'].get('achita_wg_choices', {}):
        report['skipped'].append('Parent selected a different WG override: ' + layer + str(m))
        return
    src = torch.randn((m, k), device=dev, generator=gen).to(torch.bfloat16)
    a, b = torch.empty_like(src), torch.empty_like(src)
    def run(on):
        produced = b if on else a
        _d16_spin(src, produced)
        out = ((_p16_project(produced, diag) if layer == 'o' else _p16_unpool(produced, diag))
               if on else _d16_reference(produced, layer, diag))
        return produced, out
    def mutate(seed):
        src.normal_(generator=gen)
        if seed == 1:
            src.zero_()
        elif seed == 2:
            src.mul_(16.)
    label = layer + '_M' + str(m)
    arms = {'parent': lambda: run(False), 'early': lambda: run(True)}
    same = _pdli_probe(arms, mutate, label='D16_' + label)
    row = dict(m=m, layer=layer, byte_equal=bool(same), producer='one CTA / 5000ns / first column written late',
               measurement='optimistic synthetic-producer ceiling, not a real layer predecessor')
    report['projections'].append(row)
    print('D16_BYTES', label, same, flush=True)
    if same:
        mutate(0)
        _d16_times(label, arms, dev, start, row)


def _d16_norm_fpr_probe(diag, m, gen, dev, start, report):
    if not _okh_budget(start, 274.):
        report['budget_stop'] = True
        return
    if diag.get('fpr') is None:
        report['skipped'].append('FPR ineligible M' + str(m))
        return
    branch = torch.randn((m, 8192), device=dev, generator=gen).to(torch.bfloat16)
    stream = torch.randn((m, 8192), device=dev, generator=gen)
    p = dict(ready=True, pre=(1. + .05 * torch.randn((8192,), device=dev, generator=gen)).to(torch.bfloat16),
             alpha=.1 * torch.randn((8192,), device=dev, generator=gen),
             beta=.1 * torch.randn((8192,), device=dev, generator=gen), post_eps=1e-8, pre_eps=1e-5)
    local = dict(diag, pdln=p)
    def run(mode):
        norm = (_pdln_boundary(stream, branch, local) if mode == 'early' else
                _D16_PUBLIC.fused_norm_residual(branch, stream, p['post_eps'], p['pre_eps'],
                    pre_weight=p['pre'], alpha=p['alpha'], beta=p['beta'], want_router=False, pre_norm_bf16=False))
        out = (_fpr_bfp_fused_pooler_router(norm[0], local['fpr']) if mode == 'parent'
               else _p16_pool_route(norm[0], local))
        return tuple(norm[:2]) + tuple(out)
    def mutate(seed):
        branch.normal_(generator=gen); stream.normal_(generator=gen)
        if seed == 1:
            branch.zero_(); stream.zero_()
        elif seed == 2:
            branch.mul_(16.); stream.mul_(.001)
    arms = {'parent': lambda: run('parent'), 'late': lambda: run('late'), 'early': lambda: run('early')}
    same = _pdli_probe(arms, mutate, label='D16_norm_fpr_M' + str(m))
    row = dict(m=m, byte_equal=bool(same), parent_mode=0, fixed_PDL_depth=5 if m == 16 else 3,
        late='original norm / incoming-PDL FPR', early='early norm / same incoming-PDL FPR')
    report['norm_fpr'].append(row)
    print('D16_BYTES norm_fpr', m, same, flush=True)
    if same:
        mutate(0)
        _d16_times('norm_fpr_M' + str(m), arms, dev, start, row)


def prepare(weights):
    start = _okh_clock.perf_counter()
    state = _d16_parent_prepare(weights)
    for thread in _d16_threads:
        thread.join()
    report = dict(timed_path='unchanged Dense parent', enables_nothing=True, projections=[], norm_fpr=[],
                  skipped=[], errors=[], budget_stop=False, resources={})
    try:
        dev = weights['router'].device
        prop = torch.cuda.get_device_properties(dev)
        if prop.major != 9 or prop.multi_processor_count < 129:
            raise RuntimeError('D16 unsupported GPU: ' + prop.name)
        norm_ready = _D16_PUBLIC is not None and _PDLN_EXT is not None
        if norm_ready:
            norm_resources = [int(_PDLN_EXT.query(i)) for i in range(4)]
            report['resources']['norm'] = norm_resources
            norm_ready = norm_resources[0] > 0 and norm_resources[1] == 0 and norm_resources[3] > 0
            if not norm_ready:
                report['errors'].append('D16 norm resource rejection: ' + str(norm_resources))
        else:
            report['errors'].append('D16 norm build unavailable: ' + str((_D16_PUBLIC_ERROR, _PDLN_ERROR)))
        if not _okh_budget(start, 274.):
            report['budget_stop'] = True
        else:
            fpr = None
            if norm_ready and _D16_FPR_ERROR is None:
                fpr = _fpr_prepare(weights)
            elif _D16_FPR_ERROR is not None:
                report['errors'].append('D16 FPR build unavailable: ' + str(_D16_FPR_ERROR))
            diag = dict(achita_dense=state, p16=dict(plans={}, selected={}), fpr=fpr)
            report['parent_WG_override_keys'] = sorted(map(str, state.get('achita_wg_choices', {})))
            if _P16_EXT is not None:
                _p16_plans(diag)
            else:
                report['errors'].append('D16 projection build unavailable: ' + str(_D16_BUILD_ERROR))
            report['plans'] = sorted(map(str, diag['p16']['plans']))
            if fpr is not None:
                for m in (16, 32):
                    attrs = [int(_fpr_get_ext().d16_fpr_query(m, what)) for what in range(4)]
                    report['resources']['fpr' + str(m)] = attrs
                    if attrs[0] <= 0 or attrs[1] != 0 or attrs[3] < (416 if m == 16 else 384):
                        report['errors'].append('D16 FPR resource rejection: ' + str((m, attrs)))
                        diag['fpr'] = None
            print('D16_RESOURCES', report['resources'], 'plans=', report['plans'], flush=True)
            gen = torch.Generator(device=dev).manual_seed(202610302)
            for m in (16, 32):
                for layer in ('o', 'unpooler'):
                    _d16_projection_probe(diag, m, layer, gen, dev, start, report)
                _d16_norm_fpr_probe(diag, m, gen, dev, start, report)
            torch.cuda.synchronize()
    except Exception as e:
        report['errors'].append(repr(e)[-2200:])
        print('D16_ERROR', report['errors'][-1], flush=True)
    report['prepare_s'] = _okh_clock.perf_counter() - start
    import base64, zlib, json
    payload = base64.b64encode(zlib.compress(json.dumps(report, allow_nan=False).encode(), 9)).decode()
    assert len(payload) < 9000
    print('D16_SUMMARY_Z64', payload, flush=True)
    print('D16_DONE', report['prepare_s'], 'enables_nothing=True', flush=True)
    return state
