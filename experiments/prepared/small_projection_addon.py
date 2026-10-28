"""Curated SK2 small BF16 projections; weights remain lossless and static."""
import time as _sp_time


def _sp_module():
    global _sp_ext
    if _sp_ext is None:
        _sp_ext = _fpr_load_inline(
            name='okc_small_sk2_' + hashlib.sha256(_SP_CUDA.encode()).hexdigest()[:16],
            cpp_sources=_SP_CPP, cuda_sources=_SP_CUDA, functions=['sk2small'],
            extra_cuda_cflags=['-O3', '-gencode=arch=compute_90a,code=sm_90a'],
            no_implicit_headers=True, verbose=False)
    return _sp_ext


def _sp_pack(weight, config):
    ci, layer, n, k, rows, nt, nwr, wsk, upw, nacc, stages = config
    weight = weight[:n].contiguous()
    if weight.dtype != torch.bfloat16 or tuple(weight.shape) != (n, k):
        return None
    if bool((weight.view(torch.int16) == -32768).any()) or not bool(torch.isfinite(weight).all()):
        return None
    compressed = _fpr_bfp_sk2_compress(weight)
    if compressed is None:
        return None
    table, pmax = _fpr_bfp_sk2_table(compressed, n, rows)
    sst = ((wsk * upw * 8 * 128 + nwr * wsk * upw * nt * 768) + 1023) // 1024 * 1024
    nesc = min(4, stages)
    smem = 1024 + stages * sst + 16 * stages + 32 + nesc * ((16 * (rows + 1) + 3) // 4 * 4) * 4 + pmax * 8 + 128
    if smem > 232448:
        return None
    packed = _fpr_bfp_sk2_stream(compressed['packed'], n, rows, nt, nwr, wsk, upw)
    return {'ci': ci, 'layer': layer, 'n': n, 'k': k, 'packed': packed,
            'table': table, 'pmax': pmax, 'base4': compressed['base4'], 'smem': smem}


def _sp_run(x, plan):
    out = torch.empty((x.shape[0], plan['n']), dtype=torch.bfloat16, device=x.device)
    _sp_module().sk2small(plan['packed'], x, out, plan['table'], plan['pmax'],
                         plan['base4'], x.shape[0], plan['ci'])
    return out


def _sp_select(weights, state, started):
    dev = weights['pooler'].device
    if torch.cuda.get_device_capability(dev) != (9, 0) or _sp_time.monotonic() - started > 230:
        print('SMALL_PROJ_SKIP_BUDGET', flush=True)
        return {}
    plans = [_sp_pack(weights[c[1]], c) for c in _SP_CONFIGS]
    plans = [p for p in plans if p is not None]
    if not plans or _sp_time.monotonic() - started > 235:
        return {}
    _sp_module()
    choices = {}
    flush_buffer = torch.zeros(32 * 1024 * 1024, dtype=torch.float32, device=dev)
    stream = torch.cuda.Stream(device=dev)
    stream.wait_stream(torch.cuda.current_stream(dev))
    with torch.cuda.stream(stream):
        for plan in plans:
            layer = plan['layer']
            for rows in (1, 8):
                timings = []
                safe = True
                for pattern in range(2):
                    if _sp_time.monotonic() - started > 275:
                        safe = False
                        break
                    generator = torch.Generator(device='cpu').manual_seed(20261028 + plan['ci'] * 100 + rows * 3 + pattern)
                    xcpu = torch.randn((rows, plan['k']), generator=generator)
                    if pattern:
                        xcpu[:, ::127] *= 32
                    x = xcpu.to(torch.bfloat16).to(dev)
                    sliced = UNPOOL_SLICE if layer == 'unpooler' else None

                    def call(arm):
                        return (_sp_incumbent_project(x, layer, state, sliced) if arm == 0
                                else _sp_run(x, plan))

                    for _ in range(2):
                        ref = call(0)
                        out = call(1)
                    stream.synchronize()
                    error = (torch.linalg.vector_norm(out.float() - ref.float()) /
                             torch.linalg.vector_norm(ref.float()).clamp_min(1e-20)).item()
                    if not torch.isfinite(out).all().item() or error > 0.005:
                        safe = False
                        print('SMALL_PROJ_NUMERIC_REJECT', layer, rows, pattern, error, flush=True)
                        break
                    graphs = [torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()]
                    captured_outputs = []
                    for arm in range(2):
                        with torch.cuda.graph(graphs[arm], stream=stream):
                            captured_outputs.append(call(arm))
                    samples = [[], []]
                    for repeat in range(7):
                        for arm in ((0, 1) if repeat % 2 == 0 else (1, 0)):
                            flush_buffer.add_(1)
                            begin = torch.cuda.Event(enable_timing=True)
                            end = torch.cuda.Event(enable_timing=True)
                            begin.record(stream)
                            graphs[arm].replay()
                            end.record(stream)
                            end.synchronize()
                            samples[arm].append(begin.elapsed_time(end) * 1000.0)
                    medians = [sorted(s)[len(s) // 2] for s in samples]
                    timings.append(medians)
                    print('SMALL_PROJ_PATTERN', layer, rows, pattern, medians[0], medians[1], error, flush=True)
                    del graphs, captured_outputs, ref, out, x, xcpu
                if safe and len(timings) == 2:
                    ratio = sum(t[1] for t in timings) / sum(t[0] for t in timings)
                    worst = max(t[1] / t[0] for t in timings)
                    if ratio < 0.97 and worst <= 1.02:
                        choices[layer, rows] = plan
                    print('SMALL_PROJ_DECISION', layer, rows, ratio, worst,
                          'sk2' if (layer, rows) in choices else 'incumbent', flush=True)
    torch.cuda.current_stream(dev).wait_stream(stream)
    stream.synchronize()
    print('SMALL_PROJ_CHOICES', tuple(sorted(choices)), flush=True)
    return choices


_sp_incumbent_project = _achita_dense_project


def _achita_dense_project(x, layer, state, rows=None):
    m = x.shape[0]
    plan = state.get('small_projection_choices', {}).get((layer, 1 if m == 1 else 8)) if 0 < m <= 8 else None
    if plan is not None:
        target = state['dense'][layer].shape[0] if rows is None else rows
        if (target == plan['n'] and x.shape[1] == plan['k'] and x.dtype == torch.bfloat16
                and x.is_contiguous() and x.device == plan['packed'].device and x.data_ptr() % 16 == 0):
            return _sp_run(x, plan)
    return _sp_incumbent_project(x, layer, state, rows)


_sp_original_prepare = prepare


def prepare(weights):
    started = _sp_time.monotonic()
    state = _sp_original_prepare(weights)
    state['small_projection_choices'] = _sp_select(weights, state, started)
    return state
