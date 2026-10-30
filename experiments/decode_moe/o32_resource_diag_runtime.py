"""Prepare-only completion of the previously unexercised fixed O32 question."""

_P16_EXT = None
_O32R_ERROR = None


def _o32r_build():
    global _P16_EXT, _O32R_ERROR
    try:
        assert hashlib.sha256(_O32R_CUDA.encode()).hexdigest() == _O32R_CUDA_SHA
        _P16_EXT = load_inline('okc_o32r_' + _O32R_CUDA_SHA[:12], _O32R_CPP,
            cuda_sources=_O32R_CUDA, functions=_O32R_FUNCTIONS,
            extra_cuda_cflags=['-O3', '-lineinfo', '-gencode=arch=compute_90a,code=sm_90a'],
            no_implicit_headers=True, with_pytorch_error_handling=False, verbose=False)
    except Exception as e:
        _O32R_ERROR = repr(e)[-1800:]
        print('O32R_BUILD_ERROR', _O32R_ERROR, flush=True)


_o32r_thread = _okh_bg(_o32r_build)
_d16_parent_prepare = prepare
_d16_parent_kernel_fn = kernel_fn


def _p16_project(x, state):
    if (x.shape != (32, 1024) or x.dtype != torch.bfloat16 or not x.is_contiguous()
            or x.data_ptr() % 16 or _P16_EXT is None):
        raise RuntimeError('O32R unsupported private projection')
    pp = state['p16']['plans'][32]
    out = torch.empty((32, 8192), dtype=torch.bfloat16, device=x.device)
    rc = _P16_EXT.o32r_launch(0, x.data_ptr(), pp['packed'].data_ptr(), out.data_ptr(),
                             32, torch.cuda.current_stream().cuda_stream, 0)
    if rc:
        raise RuntimeError('O32R launch: ' + str(rc))
    return out


def prepare(weights):
    start = _okh_clock.perf_counter()
    state = _d16_parent_prepare(weights)
    _o32r_thread.join()
    report = dict(timed_path='unchanged Dense parent', enables_nothing=True,
                  projections=[], skipped=[], errors=[], budget_stop=False, resources={})
    try:
        dev = weights['router'].device
        prop = torch.cuda.get_device_properties(dev)
        if prop.major != 9 or prop.multi_processor_count < 129:
            raise RuntimeError('O32R unsupported GPU: ' + prop.name)
        if _P16_EXT is None:
            raise RuntimeError('O32R module unavailable: ' + str(_O32R_ERROR))
        if not _okh_budget(start, 274.):
            report['budget_stop'] = True
        else:
            attrs = lambda: [int(_P16_EXT.o32r_attr(i)) for i in range(9)]
            report['resource_fields'] = ['regs', 'local_bytes', 'static_shared_bytes',
                'max_dynamic_shared_bytes', 'max_threads', 'preferred_carveout',
                'blocks_per_SM', 'device_shared_per_SM', 'device_shared_optin_per_block']
            report['resources']['before'] = attrs()
            report['legacy_query'] = int(_P16_EXT.o32r_legacy_query())
            report['resources']['legacy'] = attrs()
            report['private_query'] = int(_P16_EXT.o32r_launch(0, 0, 0, 0, 32, 0, 1))
            fixed = attrs()
            report['resources']['private'] = fixed
            print('O32R_RESOURCES', report['resources'], flush=True)
            bundle = _r2_find(state, 'bundle2') or {}
            ol = bundle.get('o', {})
            cfg = (32, 128, 8, 1, 1, 1)
            ready = (report['private_query'] > 0 and report['private_query'] % 10000 == 0
                and fixed[0] > 0 and fixed[1] == 0 and fixed[3] == 196736
                and fixed[4] >= 128 and fixed[6] > 0
                and fixed[2] + 196736 <= fixed[8]
                and ol.get('enabled') and cfg in ol.get('configs', ())
                and ol.get('packed') is not None)
            report['private_ready'] = bool(ready)
            diag = dict(achita_dense=state, p16=dict(plans={32: ol} if ready else {}))
            if not ready:
                report['skipped'].append('O32 private resource/configuration gate rejected')
            else:
                gen = torch.Generator(device=dev).manual_seed(202610302)
                _d16_projection_probe(diag, 32, 'o', gen, dev, start, report)
            torch.cuda.synchronize()
    except Exception as e:
        report['errors'].append(repr(e)[-2200:])
        print('O32R_ERROR', report['errors'][-1], flush=True)
    report['prepare_s'] = _okh_clock.perf_counter() - start
    import base64, zlib, json
    payload = base64.b64encode(zlib.compress(json.dumps(report, allow_nan=False).encode(), 9)).decode()
    assert len(payload) < 9000
    print('O32R_SUMMARY_Z64', payload, flush=True)
    print('O32R_DONE', report['prepare_s'], 'enables_nothing=True', flush=True)
    return state
