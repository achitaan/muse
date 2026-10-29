# Fixed tiny Decode-tail experiment. All inherited source notices are retained.
import json as _dt_json
import statistics as _dt_stats
from torch.utils.cpp_extension import load_inline as _dt_load

_dt_old_prepare = prepare
_dt_modules = {}


def _dt_ext(defines):
    defines = tuple(defines)
    if defines not in _dt_modules:
        tag = '_'.join(d.removeprefix('-D').replace('=', '') for d in defines)
        _dt_modules[defines] = _dt_load(
            name='okc_decoder_tail_SOURCE_TAG_' + tag,
            cpp_sources=_DT_CPP, cuda_sources=_DT_CUDA,
            functions=['okc_run_tail', 'okc_tail_resources'],
            extra_cuda_cflags=['-O3', '-gencode=arch=compute_90a,code=sm_90a',
                               '-lineinfo', '-DOKC_NS=3', '-DOKC_NSF=3', '-DOKC_DQ7=1', *defines],
            verbose=False)
    return _dt_modules[defines]


def _dt_pack(weight):
    assert weight.dtype == torch.bfloat16 and weight.shape[1] == 4096
    dev = weight.device
    rb = torch.arange(64, device=dev).reshape(64, 1, 1, 1, 1, 1)
    ki = torch.arange(32, device=dev).reshape(1, 32, 1, 1, 1, 1)
    warp = torch.arange(8, device=dev).reshape(1, 1, 8, 1, 1, 1)
    lane = torch.arange(32, device=dev).reshape(1, 1, 1, 32, 1, 1)
    word = torch.arange(4, device=dev).reshape(1, 1, 1, 1, 4, 1)
    half = torch.arange(2, device=dev).reshape(1, 1, 1, 1, 1, 2)
    row = rb * 16 + (lane >> 2) + (word & 1) * 8
    col = warp * 512 + ki * 16 + (lane & 3) * 2 + (word >> 1) * 8 + half
    indices = row * 4096 + col
    bits = weight[:1024].contiguous().view(torch.int16).reshape(-1)
    packed = bits[indices].contiguous()
    inverse = torch.empty_like(bits)
    inverse[indices.reshape(-1)] = packed.reshape(-1)
    if not torch.equal(inverse, bits):
        raise RuntimeError('Decode tail BF16 weight permutation is not exact')
    return packed.view(torch.bfloat16).reshape(-1)


def _dt_call(x, tw, ids, stream, state):
    st = state['decoder_tail']
    m = x.shape[0]
    moe = torch.empty((m, 4096), dtype=torch.bfloat16, device=x.device)
    branch = torch.empty((m, 1024), dtype=torch.bfloat16, device=x.device)
    output = torch.empty((m, 8192), dtype=torch.bfloat16, device=x.device)
    updated = torch.empty((m, 8192), dtype=torch.float32, device=x.device)
    cache = state['sliced_ffn_boundary']
    tail = [st['weight'], branch, stream, cache['post_weight'], cache['pre_weight'],
            cache['alpha'], cache['beta'], output, updated]
    ext = st['modules'][0 if m <= 4 else 1]
    native = state['native_moe']
    ext.okc_run_tail(native['w'], st['scratch'], x, ids, tw, moe,
                    native['lo'], native['cnt'], native['grid'], tail)
    return output, updated, moe, branch


def _dt_original(x, tw, ids, stream, state):
    moe = _experts(x, tw, ids, state)
    branch = _achita_dense_project(moe, 'unpooler', state, UNPOOL_SLICE)
    output, updated = _sfn_boundary(stream, branch, state)
    return output, updated, moe, branch


def _dt_error(actual, expected, label):
    result = []
    for index, (a, b) in enumerate(zip(actual, expected)):
        if a.shape != b.shape or a.dtype != b.dtype or not bool(torch.isfinite(a).all()):
            raise RuntimeError(f'Decode tail {label} shape/dtype/finite {index}')
        av, bv = a.float(), b.float()
        diff = torch.linalg.vector_norm(av - bv, dim=-1)
        den = torch.linalg.vector_norm(bv, dim=-1)
        floor = max(float(den.mean()) * .1, 1e-12)
        g = float(torch.linalg.vector_norm(av - bv) / torch.linalg.vector_norm(bv).clamp_min(1e-12))
        r = float((diff / den.clamp_min(floor)).max())
        # This design adds an exact-BF16 GEMM with different FP32 partitioning;
        # unchanged-boundary-only experiments used separate, tighter gates.
        limits = (.001, .003) if index == 2 else (.002, .004)
        if g > limits[0] or r > limits[1]:
            raise RuntimeError(f'Decode tail {label} numerical {index}: {g},{r}')
        result.append([g, r])
    return result


def _dt_fixture(m, dev, seed, kind='normal'):
    gen = torch.Generator(device=dev).manual_seed(seed)
    x = torch.randn((m, 4096), generator=gen, device=dev, dtype=torch.bfloat16)
    stream = torch.randn((m, 8192), generator=gen, device=dev, dtype=torch.float32)
    # Include repeated local routes: the parent supports them, so the tail must.
    ids = torch.randint(32, 256, (m, 8), generator=gen, device=dev, dtype=torch.int32)
    ids[:, :2] = torch.arange(m, device=dev, dtype=torch.int32)[:, None] % (2 if seed & 1 else 7)
    if kind == 'all_nonlocal': ids[:] = 127
    if kind == 'duplicate': ids[:, :5] = 3
    if kind == 'outlier': x[:, 17] = 16; stream[:, 34] = -32
    tw = torch.rand((m, 8), generator=gen, device=dev, dtype=torch.float32)
    tw = (tw / tw.sum(-1, keepdim=True)).contiguous()
    return x, tw, ids, stream


def _dt_paired(old, new, dev, seed):
    flush = torch.ones((128, 1024, 1024), dtype=torch.float32, device=dev)
    sink = torch.empty((), dtype=torch.float32, device=dev)
    stream = torch.cuda.Stream(device=dev)
    stream.wait_stream(torch.cuda.current_stream(dev))
    with torch.cuda.stream(stream):
        old(); new(); stream.synchronize()
        graphs = [torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()]
        for arm, call in enumerate((old, new)):
            with torch.cuda.graph(graphs[arm], stream=stream): call()
        samples, events = [], []
        for pair in range(10):
            order = (0, 1) if (pair + seed) % 2 == 0 else (1, 0)
            samples.append(dict(pair=pair, order=list(order)))
            for arm in order:
                torch.sum(flush, dim=(0, 1, 2), out=sink)
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record(stream); graphs[arm].replay(); b.record(stream)
                events.append((pair, arm, a, b))
        stream.synchronize()
    for pair, arm, a, b in events: samples[pair][arm] = a.elapsed_time(b) * 1_000_000
    ratios = [r[1] / r[0] for r in samples]
    savings = [r[0] - r[1] for r in samples]
    return dict(new_over_old=_dt_stats.median(ratios), saved_ns=_dt_stats.median(savings),
                sample_schema=['pair', 'first_arm', 'old_ns', 'new_ns'],
                samples=[[r['pair'], r['order'][0], r[0], r[1]] for r in samples])


def _dt_emit(report):
    payload = _dt_json.dumps(report, separators=(',', ':'))
    if len(payload.encode()) > 14500:
        raise RuntimeError('Decode tail compact diagnostic bound exceeded')
    print('DECODER_TAIL ' + payload, flush=True)


def _dt_prepare(weights, state):
    dev = weights['unpooler'].device
    native = state['native_moe']
    cache = state['sliced_ffn_boundary']
    report = dict(source_tag='SOURCE_TAG', cuda='CUDA_TAG', selected=False, resources=[],
                  numerical=[], comparison_maxima=[[0.,0.] for _ in range(4)],
                  paired=[], replay=[], weight_permutation_bitwise=False)
    state['_decoder_tail_report'] = report
    state['decoder_tail'] = None
    if (torch.cuda.get_device_capability(dev)[0] != 9 or cache is None
            or native['sfx'] != ('-DOKC_PRMTSC=1',)
            or cache['post_eps'] != 1e-8 or cache['pre_eps'] != 1e-5):
        _dt_emit(report)
        return
    for key in ('post_weight', 'pre_weight', 'alpha', 'beta'):
        value = cache[key]
        dtype = torch.float32 if key in ('alpha', 'beta') else torch.bfloat16
        if value.dtype != dtype or not value.is_contiguous() or value.numel() != 8192:
            raise RuntimeError('Decode tail unsupported norm cache ' + key)
    modules = [_dt_ext(tuple(d) + native['sfx']) for d in (_native_moe_TINY_DEFINES, _native_moe_TINY2_DEFINES)]
    for family, ext in enumerate(modules):
        resource = list(ext.okc_tail_resources())
        if len(resource) != 28: raise RuntimeError('Decode tail resource schema')
        for start in (0, 14):
            q = resource[start:start+14]
            if (q[1:6] != [0]*5 or not (0 < q[6] <= 160) or q[7] != 0
                    or not (0 <= q[8] <= 1024) or q[9] + q[8] > 232448
                    or q[10] != native['grid'] or q[11] < 1 or q[12] != 1 or q[13] != 288
                    or native['grid'] > q[10] * q[11]):
                raise RuntimeError('Decode tail native resources inadmissible ' + str(q))
        report['resources'].append(dict(family=family, values=resource))
    state['decoder_tail'] = dict(modules=modules, weight=_dt_pack(weights['unpooler']),
                                 scratch=_native_moe__make_scratch(dev), selected=False)
    report['weight_permutation_bitwise'] = True
    for m in (1, 2, 3, 4, 7, 8):
        for kind in ('normal', 'all_nonlocal', 'duplicate', 'outlier'):
            args = _dt_fixture(m, dev, 20261029 + m, kind)
            new = _dt_call(*args, state)
            old = _dt_original(*args, state)
            error = _dt_error(new, old, f'{m}/{kind}')
            # Independent FP64 dense oracle uses the actual new BF16 MoE
            # output, so expert or norm differences cannot hide an unpool error.
            oracle = (new[2].double() @ weights['unpooler'][:1024].double().T).to(torch.bfloat16)
            ag, bg = new[3].float(), oracle.float()
            den = torch.linalg.vector_norm(bg, dim=-1)
            floor = max(float(den.mean())*.1, 1e-12)
            g = float(torch.linalg.vector_norm(ag-bg) / torch.linalg.vector_norm(bg).clamp_min(1e-12))
            r = float((torch.linalg.vector_norm(ag-bg,dim=-1) / den.clamp_min(floor)).max())
            if g > .0005 or r > .002: raise RuntimeError(f'Decode tail FP64 dense oracle {m}/{kind}: {g},{r}')
            if kind == 'all_nonlocal' and (bool(torch.count_nonzero(new[2])) or bool(torch.count_nonzero(new[3]))):
                raise RuntimeError('Decode tail all-nonlocal output is nonzero')
            for index, pair in enumerate(error):
                report['comparison_maxima'][index] = [max(a,b) for a,b in zip(report['comparison_maxima'][index],pair)]
            report['numerical'].append(dict(m=m, kind=kind, dense_oracle=[g,r]))
    # Distinct graphs share only this backend's private sequential scratch.
    # The retained backend has separate scratch, including when interleaved.
    graphs, inputs, outputs = [], [], []
    for m in (1, 8, 3, 7, 2, 4, 8, 1):
        args = _dt_fixture(m, dev, 20300000+m)
        _dt_call(*args, state)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph): value = _dt_call(*args, state)
        graphs.append(graph); inputs.append(args); outputs.append(value)
    for cycle in range(2):
        for index, (graph, args, value) in enumerate(zip(graphs, inputs, outputs)):
            fresh = _dt_fixture(args[0].shape[0], dev, 20400000+cycle*100+index,
                                'duplicate' if cycle else 'normal')
            for a, b in zip(args, fresh): a.copy_(b)
            graph.replay()
            old = _dt_original(*args, state)
            error = _dt_error(value, old, f'replay/{cycle}/{index}')
            ready = state['decoder_tail']['scratch'][0][8+768:8+776]
            if bool(torch.count_nonzero(ready)): raise RuntimeError('Decode tail stale readiness after replay')
            report['replay'].append(dict(cycle=cycle, index=index, m=args[0].shape[0],
                                         max_error=[max(p[i] for p in error) for i in range(2)]))
    for m in (1, 8, 3, 7):
        for seed in (20261029, 20261030):
            args = _dt_fixture(m, dev, seed)
            paired = _dt_paired(lambda: _dt_original(*args, state), lambda: _dt_call(*args, state), dev, seed)
            report['paired'].append(dict(m=m, seed=seed, **paired))
    changed = [r for r in report['paired'] if r['m'] in (1, 8)]
    nearby = [r for r in report['paired'] if r['m'] not in (1, 8)]
    # Absolute saving screen: ~4.6 us at both affected six-load entries is
    # needed for a plausible 1.5% whole-layer gain. Full-layer guards remain.
    select = (all(r['new_over_old'] < .97 and r['saved_ns'] >= 4600 for r in changed)
              and all(r['new_over_old'] <= 1.02 for r in nearby))
    report['selected'] = select
    state['decoder_tail']['selected'] = select
    _dt_emit(report)
    state['_decoder_tail_report_printed'] = True
    if not select:
        # A losing setup experiment terminates without timing the old fallback
        # and presenting that fallback's score as evidence for the new kernel.
        raise RuntimeError('Decode tail rejected by frozen complete-tail performance screen')


def _dt_eligible(x, stream, state):
    st = state.get('decoder_tail')
    return (st is not None and st['selected'] and 1 <= x.shape[0] <= 8
            and x.is_contiguous() and stream.is_contiguous())


def prepare(weights):
    state = _dt_old_prepare(weights)
    try:
        _dt_prepare(weights, state)
    except Exception as exc:
        if not state.get('_decoder_tail_report_printed'):
            report = state.get('_decoder_tail_report', {})
            report['setup_error'] = str(exc)
            _dt_emit(report)
        raise
    torch.cuda.synchronize()
    return state
