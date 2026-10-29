"""Two-CTA activation multicast; retained small and ineligible paths stay exact."""
import hashlib as _cx_hash
import json as _cx_json
import math as _cx_math
import random as _cx_random
import statistics as _cx_stats
import time as _cx_time
from torch.utils.cpp_extension import load_inline as _cx_load

_cx_original_prepare = prepare
_cx_original_kernel = kernel_fn
_CLUSTER_SHAPES = (1, 4, 8, 16, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 384, 385)


def _cx_module(state):
    assert _cx_hash.sha256(_CLUSTER_CUDA.encode()).hexdigest() == '2701a25e854caef3a46c1ae2002bd59932b2b056a05a1c0d2cf498dc5f8b6114'
    defs = list(_main_defines(state['sfx'])) + list(state['sfx'])
    if DQ7 and '-DOKC_DQ7=1' not in defs:
        defs.append('-DOKC_DQ7=1')
    tag = _cx_hash.sha256((_CLUSTER_CUDA + repr(defs)).encode()).hexdigest()[:16]
    module = _cx_load(name='okc_moe_cluster2_' + tag,
                      cpp_sources=CPP_SRC + '\nstd::vector<int64_t> okc_cluster2_info(int64_t grid);\n',
                      cuda_sources=_CLUSTER_CUDA,
                      functions=['okc_run', 'okc_set_pdl', 'okc_cluster2_info'],
                      extra_cuda_cflags=['-O3', '-gencode=arch=compute_90a,code=sm_90a', '-lineinfo',
                                        '-DOKC_NS=3', '-DOKC_NSF=3', *defs], verbose=False)
    module.okc_set_pdl(0)
    return module


def _cx_resources_ok(r, grid):
    return (len(r) == 21 and all(isinstance(v, int) for v in r)
            and all(0 < v <= 112 for v in r[:2]) and r[2:6] == [0, 0, 0, 0]
            and r[6:10] == [231168, 576, 1, 1]
            and isinstance(grid, int) and 0 < grid <= 160 and grid % 2 == 0
            and r[10] >= grid // 2 and r[11] >= grid // 2
            and r[12:18] == [2, 1, 1, 2, 1, 1] and r[18:] == [grid, 1, 20261029])


def _cx_input(m, dev, mode='normal', replay=0):
    seed = 20261030 + m + 100000 * replay + (10000 if mode == 'skewed' else 0)
    generator = torch.Generator(device=dev).manual_seed(seed)
    x = torch.randn((m, HIDDEN), generator=generator, device=dev).bfloat16()
    scores = torch.rand((m, 256), generator=generator, device=dev)
    if mode == 'skewed':
        scores = scores + torch.rand((256,), generator=generator, device=dev).pow(6) * 4
    else:
        z = torch.randn((256,), generator=generator, device=dev)
        u = torch.rand((256,), generator=generator, device=dev).clamp_min(1e-12)
        scores = scores + .03 * z / (-u.log()).sqrt()
    values, ids = scores.topk(8, dim=1)
    counts = torch.bincount((ids // 32).flatten(), minlength=8)
    ids = ((ids - 32 * counts.argmax()) % 256).to(torch.int32).contiguous()
    if mode == 'remote':
        ids = torch.arange(32, 40, dtype=torch.int32, device=dev).expand(m, 8).contiguous()
    elif mode == 'duplicate':
        ids.zero_()
    weights = (values / values.sum(1, keepdim=True)).float().contiguous()
    return dict(hidden_states=x, topk_ids=ids, topk_weights=weights,
                config=dict(local_start=0, local_count=32))


def _cx_run(data, state, module, scratch):
    cfg = data['config']
    assert cfg['local_start'] == state['lo'] and cfg['local_count'] == state['cnt']
    x = data['hidden_states'].contiguous()
    ids = data['topk_ids'].to(torch.int32).contiguous()
    if ids.data_ptr() & 15:
        ids = ids.clone()
    tw = data['topk_weights'].to(torch.float32).contiguous()
    m = x.shape[0]
    out = torch.empty((m, HIDDEN), dtype=torch.bfloat16, device=x.device)
    for first in range(0, m, CHUNK_TOKENS):
        last = min(m, first + CHUNK_TOKENS)
        module.okc_run(state['w'], scratch, x[first:last], ids[first:last], tw[first:last], out[first:last],
                       state['lo'], state['cnt'], state['grid'])
    return out


def _cx_error(reference, got):
    reference, got = reference.float(), got.float()
    assert bool(torch.isfinite(reference).all().item()) and bool(torch.isfinite(got).all().item())
    den = torch.linalg.vector_norm(reference, dim=1)
    relative = torch.linalg.vector_norm(got - reference) / torch.linalg.vector_norm(reference).clamp_min(1e-12)
    worst = torch.linalg.vector_norm(got - reference, dim=1) / den.clamp_min(.1 * den.mean()).clamp_min(1e-12)
    relative, worst = float(relative.item()), float(worst.max().item())
    assert relative <= .005 and worst <= .02
    return relative, worst


def _cx_poison(scratch):
    scratch[1][EMAX:EMAX + EMAX * HIDDEN].view(torch.bfloat16).fill_(float('nan'))


def _cx_case(m, mode, state, module, scratch, flush, sink, stream):
    print('CLUSTER_X_CASE ' + str(m) + ' ' + mode, flush=True)
    dev = state['w'][0].device
    data = _cx_input(m, dev, mode)
    outputs = [None, None]

    def call(arm):
        outputs[arm] = (_cx_original_kernel(data, state) if arm == 0 else
                        _cx_run(data, state, module, scratch))

    stream.wait_stream(torch.cuda.current_stream(dev))
    with torch.cuda.stream(stream):
        _cx_poison(state['scratch']); _cx_poison(scratch)
        for arm in (0, 1):
            call(arm)
        stream.synchronize()
        graphs = [torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()]
        for arm in (0, 1):
            with torch.cuda.graph(graphs[arm], stream=stream):
                call(arm)
        orders = [(0, 1)] * 5 + [(1, 0)] * 5
        _cx_random.Random(20261030 + m + (10000 if mode == 'skewed' else 0)).shuffle(orders)
        events = []
        for repeat, order in enumerate(orders):
            for arm in order:
                torch.sum(flush, dim=0, out=sink)
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record(stream); graphs[arm].replay(); b.record(stream)
                events.append((repeat, arm, a, b))
        stream.synchronize()
        pairs = [{} for _ in orders]
        for repeat, arm, a, b in events:
            pairs[repeat][arm] = a.elapsed_time(b) * 1_000_000
        relative, worst = _cx_error(outputs[0], outputs[1])
        fresh = []
        if m in (128, 385) and mode == 'normal':
            for replay in (1, 2):
                changed = _cx_input(m, dev, mode, replay)
                for key in ('hidden_states', 'topk_ids', 'topk_weights'):
                    data[key].copy_(changed[key])
                _cx_poison(state['scratch']); _cx_poison(scratch)
                graphs[0].replay(); graphs[1].replay(); stream.synchronize()
                fresh.append(_cx_error(outputs[0], outputs[1]))
    torch.cuda.current_stream(dev).wait_stream(stream)
    ratios = [p[1] / p[0] for p in pairs]
    return dict(m=m, mode=mode, synthetic=True, seed=20261030 + m + (10000 if mode == 'skewed' else 0),
                old_ns=[p[0] for p in pairs], new_ns=[p[1] for p in pairs], arm_orders=orders,
                median_ratio=_cx_stats.median(ratios), relative=relative, worst=worst,
                changed_input_graph_replays=fresh)


def _cx_decision(cases, resources, grid, analytic):
    if [(c['m'], c['mode']) for c in cases] != [(m, 'normal') for m in _CLUSTER_SHAPES] + [(128, 'skewed')]:
        return False
    for case in cases:
        old, new = case['old_ns'], case['new_ns']
        orders = [tuple(o) for o in case['arm_orders']]
        if (len(old) != 10 or len(new) != 10
                or not all(_cx_math.isfinite(v) and v > 0 for v in old + new)
                or orders.count((0, 1)) != 5 or orders.count((1, 0)) != 5
                or not _cx_math.isclose(case['median_ratio'], _cx_stats.median(n / o for o, n in zip(old, new)), rel_tol=1e-10)
                or not 0 <= case['relative'] <= .005 or not 0 <= case['worst'] <= .02):
            return False
        if case['mode'] == 'normal' and case['m'] in (128, 385):
            fresh = case['changed_input_graph_replays']
            if len(fresh) != 2 or not all(len(p) == 2 and 0 <= p[0] <= .005 and 0 <= p[1] <= .02 for p in fresh):
                return False
    if [(a['m'], a['mode']) for a in analytic] != [(m, 'remote') for m in (64, 128, 384, 385)] + [(128, 'duplicate')]:
        return False
    if not all(a['passed'] for a in analytic):
        return False
    normal = {c['m']: c for c in cases if c['mode'] == 'normal'}
    ratios = [normal[m]['median_ratio'] for m in (128, 256)]
    return (_cx_resources_ok(resources, grid)
            and all(c['median_ratio'] <= 1.02 for c in cases if c['m'] >= 64)
            and all(r < .97 for r in ratios)
            and _cx_math.prod(ratios) ** .5 < 1 / 1.03)


def prepare(weights):
    start = _cx_time.monotonic()
    state = _cx_original_prepare(weights)
    state['_cx_enabled'] = False
    evidence = dict(cuda_sha256=_cx_hash.sha256(_CLUSTER_CUDA.encode()).hexdigest(),
                    synthetic_only=True, independent_jobs=1, repeats_are_not_independent=True,
                    selected_main=False, resources=None, cases=[], analytic=[],
                    extra_weight_layouts=0, separate_scratch=True,
                    original_small_and_ineligible_scale_paths=True, minimum_timed_M=64)
    dev = weights['w13'].device
    if not state['sfx'] or torch.cuda.get_device_capability(dev) != (9, 0) or _cx_time.monotonic() - start > 185:
        evidence['skip'] = 'eligibility or original preparation budget'
    else:
        module = _cx_module(state)
        try:
            resources = list(module.okc_cluster2_info(state['grid']))
        except RuntimeError as error:
            resources = []
            evidence['skip'] = 'cluster resource/launch-configuration API rejected in prepare: ' + str(error)[:600]
        evidence['resources'] = resources
        evidence['grid'] = state['grid']
        print('CLUSTER_X_RESOURCES ' + _cx_json.dumps(dict(cuda_sha256=evidence['cuda_sha256'], resources=resources, grid=state['grid'])), flush=True)
        if not _cx_resources_ok(resources, state['grid']):
            evidence.setdefault('skip', 'Resource/cluster residency gate failed before any new-kernel launch')
        elif _cx_time.monotonic() - start > 270:
            evidence['skip'] = 'extra compilation exhausted preparation budget'
        else:
            scratch = _make_scratch(dev)
            flush = torch.zeros((32768, 1024), dtype=torch.float32, device=dev)
            sink = torch.empty((1024,), dtype=torch.float32, device=dev)
            stream = torch.cuda.Stream(device=dev)
            for m in _CLUSTER_SHAPES:
                if _cx_time.monotonic() - start > 280:
                    break
                evidence['cases'].append(_cx_case(m, 'normal', state, module, scratch, flush, sink, stream))
            if _cx_time.monotonic() - start <= 280:
                evidence['cases'].append(_cx_case(128, 'skewed', state, module, scratch, flush, sink, stream))
                for m in (64, 128, 384, 385):
                    _cx_poison(scratch)
                    result = _cx_run(_cx_input(m, dev, 'remote'), state, module, scratch)
                    assert bool(torch.isfinite(result).all().item()) and int(torch.count_nonzero(result).item()) == 0
                    evidence['analytic'].append(dict(m=m, mode='remote', passed=True))
                data = _cx_input(128, dev, 'duplicate')
                _cx_poison(state['scratch']); _cx_poison(scratch)
                reference = _cx_original_kernel(data, state)
                got = _cx_run(data, state, module, scratch)
                relative, worst = _cx_error(reference, got)
                evidence['analytic'].append(dict(m=128, mode='duplicate', passed=True, relative=relative, worst=worst))
            if _cx_decision(evidence['cases'], resources, state['grid'], evidence['analytic']):
                state['_cx_enabled'] = evidence['selected_main'] = True
                state['_cx_module'] = module
                state['_cx_scratch'] = scratch
    evidence['prepare_s'] = _cx_time.monotonic() - start
    evidence['software'] = dict(torch=torch.__version__, cuda=torch.version.cuda, device=torch.cuda.get_device_name(dev))
    encoded = _cx_json.dumps(evidence, separators=(',', ':'))
    assert len(encoded) <= 14500
    print('CLUSTER_X_EVIDENCE ' + encoded, flush=True)
    torch.cuda.synchronize(dev)
    return state


def kernel_fn(data, state):
    if not state.get('_cx_enabled', False) or data['hidden_states'].shape[0] < 64 or not state['sfx']:
        return _cx_original_kernel(data, state)
    return _cx_run(data, state, state['_cx_module'], state['_cx_scratch'])
