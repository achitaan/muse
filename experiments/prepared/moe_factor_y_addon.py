"""One lossless WGS scratch/cache candidate; original small/fallback paths stay exact."""
import hashlib as _factor_hash
import json as _factor_json
import math as _factor_math
import random as _factor_random
import statistics as _factor_stats
import time as _factor_time
from torch.utils.cpp_extension import load_inline as _factor_load

_factor_original_prepare = prepare
_factor_original_kernel = kernel_fn
_FACTOR_SHAPES = (17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 191, 192, 255, 256, 257, 384, 385)
_FACTOR_QUERY = r'''
#include <c10/cuda/CUDAException.h>
std::vector<int64_t> okc_factor_info() {
  size_t sm=okc_smem_bytes();
  C10_CUDA_CHECK(cudaFuncSetAttribute(okc_factor_y::moe_fused<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm));
  C10_CUDA_CHECK(cudaFuncSetAttribute(okc_factor_y::moe_fused<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm));
  cudaFuncAttributes a{}, b{};
  C10_CUDA_CHECK(cudaFuncGetAttributes(&a, okc_factor_y::moe_fused<false>));
  C10_CUDA_CHECK(cudaFuncGetAttributes(&b, okc_factor_y::moe_fused<true>));
  int oa=0, ob=0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&oa, okc_factor_y::moe_fused<false>, okc_factor_y::NTL, sm));
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&ob, okc_factor_y::moe_fused<true>, okc_factor_y::NTL, sm));
  return {a.numRegs, b.numRegs, (int64_t)a.localSizeBytes, (int64_t)b.localSizeBytes,
          (int64_t)a.sharedSizeBytes, (int64_t)b.sharedSizeBytes, (int64_t)sm, oa, ob, okc_factor_y::NTL};
}
'''


def _factor_module(state):
    defs = list(_main_defines(state['sfx'])) + list(state['sfx'])
    if DQ7 and '-DOKC_DQ7=1' not in defs:
        defs.append('-DOKC_DQ7=1')
    source = _FACTOR_CUDA + _FACTOR_QUERY
    tag = _factor_hash.sha256((source + repr(defs)).encode()).hexdigest()[:16]
    module = _factor_load(name='okc_moe_factor_y_' + tag,
                          cpp_sources=CPP_SRC + '\nstd::vector<int64_t> okc_factor_info();\n',
                          cuda_sources=source, functions=['okc_run', 'okc_set_pdl', 'okc_factor_info'],
                          extra_cuda_cflags=['-O3', '-gencode=arch=compute_90a,code=sm_90a', '-lineinfo',
                                            '-DOKC_NS=3', '-DOKC_NSF=3', *defs], verbose=False)
    module.okc_set_pdl(0)
    return module


def _factor_resources_ok(resources, grid):
    return (len(resources) == 10 and all(0 < r <= 112 for r in resources[:2])
            and resources[2:6] == [0, 0, 0, 0] and resources[6] == 231168
            and resources[7] >= 1 and resources[8] >= 1 and resources[9] == 576
            and 0 < grid <= 160)


def _factor_input(m, dev, mode='normal', replay=0):
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


def _factor_run(data, state, module, scratch):
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


def _factor_error(reference, got):
    reference, got = reference.float(), got.float()
    assert bool(torch.isfinite(reference).all().item()) and bool(torch.isfinite(got).all().item())
    den = torch.linalg.vector_norm(reference, dim=1)
    relative = torch.linalg.vector_norm(got - reference) / torch.linalg.vector_norm(reference).clamp_min(1e-12)
    worst = torch.linalg.vector_norm(got - reference, dim=1) / den.clamp_min(.1 * den.mean()).clamp_min(1e-12)
    relative, worst = float(relative.item()), float(worst.max().item())
    assert relative <= .005 and worst <= .02
    return relative, worst


def _factor_poison(scratch):
    scratch[1][EMAX:EMAX + EMAX * HIDDEN].view(torch.bfloat16).fill_(float('nan'))


def _factor_case(m, mode, state, module, scratch, flush, sink, stream):
    dev = state['w'][0].device
    data = _factor_input(m, dev, mode)
    outputs = [None, None]

    def call(arm):
        outputs[arm] = (_factor_original_kernel(data, state) if arm == 0 else
                        _factor_run(data, state, module, scratch))

    stream.wait_stream(torch.cuda.current_stream(dev))
    with torch.cuda.stream(stream):
        _factor_poison(state['scratch'])
        _factor_poison(scratch)
        for arm in (0, 1):
            call(arm)
        stream.synchronize()
        graphs = [torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()]
        for arm in (0, 1):
            with torch.cuda.graph(graphs[arm], stream=stream):
                call(arm)
        orders = [(0, 1)] * 5 + [(1, 0)] * 5
        _factor_random.Random(20261030 + m + (10000 if mode == 'skewed' else 0)).shuffle(orders)
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
        relative, worst = _factor_error(outputs[0], outputs[1])
        fresh = []
        if m == 32 and mode == 'normal':
            for replay in (1, 2):
                changed = _factor_input(m, dev, mode, replay)
                for key in ('hidden_states', 'topk_ids', 'topk_weights'):
                    data[key].copy_(changed[key])
                _factor_poison(state['scratch']); _factor_poison(scratch)
                graphs[0].replay(); graphs[1].replay(); stream.synchronize()
                fresh.append(_factor_error(outputs[0], outputs[1]))
    torch.cuda.current_stream(dev).wait_stream(stream)
    ratios = [p[1] / p[0] for p in pairs]
    return dict(m=m, mode=mode, synthetic=True, seed=20261030 + m + (10000 if mode == 'skewed' else 0),
                old_ns=[p[0] for p in pairs], new_ns=[p[1] for p in pairs], arm_orders=orders,
                median_ratio=_factor_stats.median(ratios), relative=relative, worst=worst,
                changed_input_graph_replays=fresh)


def _factor_decision(cases, resources, grid, analytic):
    if [(c['m'], c['mode']) for c in cases] != [(m, 'normal') for m in _FACTOR_SHAPES] + [(128, 'skewed')]:
        return False
    for case in cases:
        old, new = case['old_ns'], case['new_ns']
        orders = [tuple(o) for o in case['arm_orders']]
        if (len(old) != 10 or len(new) != 10
                or not all(_factor_math.isfinite(v) and v > 0 for v in old + new)
                or orders.count((0, 1)) != 5 or orders.count((1, 0)) != 5
                or not _factor_math.isclose(case['median_ratio'], _factor_stats.median(n / o for o, n in zip(old, new)), rel_tol=1e-10)
                or not 0 <= case['relative'] <= .005 or not 0 <= case['worst'] <= .02):
            return False
    if len(cases[2]['changed_input_graph_replays']) != 2:
        return False
    if not all(len(pair) == 2 and 0 <= pair[0] <= .005 and 0 <= pair[1] <= .02
               for pair in cases[2]['changed_input_graph_replays']):
        return False
    if [(a['m'], a['mode']) for a in analytic] != [(m, 'remote') for m in (17, 33, 384, 385)] + [(33, 'duplicate')]:
        return False
    normal = {c['m']: c for c in cases if c['mode'] == 'normal'}
    if set(normal) != set(_FACTOR_SHAPES) or len(cases) != len(_FACTOR_SHAPES) + 1:
        return False
    ratios = [normal[m]['median_ratio'] for m in (32, 128, 256)]
    return (_factor_resources_ok(resources, grid) and len(analytic) == 5
            and all(a['passed'] for a in analytic)
            and all(c['median_ratio'] <= 1.02 for c in cases)
            and all(r < .97 for r in ratios)
            and _factor_math.prod(ratios) ** (1 / 3) < 1 / 1.03)


def prepare(weights):
    start = _factor_time.monotonic()
    state = _factor_original_prepare(weights)
    state['_factor_enabled'] = False
    evidence = dict(cuda_sha256=_factor_hash.sha256(_FACTOR_CUDA.encode()).hexdigest(),
                    synthetic_only=True, independent_jobs=1, repeats_are_not_independent=True,
                    selected_main=False, resources=None, cases=[], analytic=[],
                    extra_weight_layouts=0, separate_scratch=True,
                    original_small_and_ineligible_scale_paths=True)
    dev = weights['w13'].device
    if not state['sfx'] or torch.cuda.get_device_capability(dev) != (9, 0) or _factor_time.monotonic() - start > 185:
        evidence['skip'] = 'eligibility or original preparation budget'
    else:
        module = _factor_module(state)
        resources = module.okc_factor_info()
        evidence['resources'] = resources
        evidence['grid'] = state['grid']
        if not _factor_resources_ok(resources, state['grid']):
            evidence['skip'] = 'Resource gate failed before any new-kernel launch'
        elif _factor_time.monotonic() - start > 270:
            evidence['skip'] = 'extra compilation exhausted preparation budget'
        else:
            scratch = _make_scratch(dev)
            flush = torch.zeros((32768, 1024), dtype=torch.float32, device=dev)
            sink = torch.empty((1024,), dtype=torch.float32, device=dev)
            stream = torch.cuda.Stream(device=dev)
            for m in _FACTOR_SHAPES:
                if _factor_time.monotonic() - start > 280:
                    break
                evidence['cases'].append(_factor_case(m, 'normal', state, module, scratch, flush, sink, stream))
            if _factor_time.monotonic() - start <= 280:
                evidence['cases'].append(_factor_case(128, 'skewed', state, module, scratch, flush, sink, stream))
                # Empty-job mathematical oracle; do not run the original WGS
                # path, whose empty range reads an uninitialized job-table row.
                for m in (17, 33, 384, 385):
                    _factor_poison(scratch)
                    result = _factor_run(_factor_input(m, dev, 'remote'), state, module, scratch)
                    assert bool(torch.isfinite(result).all().item()) and int(torch.count_nonzero(result).item()) == 0
                    evidence['analytic'].append(dict(m=m, mode='remote', passed=True))
                data = _factor_input(33, dev, 'duplicate')
                _factor_poison(state['scratch']); _factor_poison(scratch)
                reference = _factor_original_kernel(data, state)
                got = _factor_run(data, state, module, scratch)
                relative, worst = _factor_error(reference, got)
                evidence['analytic'].append(dict(m=33, mode='duplicate', passed=True, relative=relative, worst=worst))
            if _factor_decision(evidence['cases'], resources, state['grid'], evidence['analytic']):
                state['_factor_enabled'] = evidence['selected_main'] = True
                state['_factor_module'] = module
                state['_factor_scratch'] = scratch
    evidence['prepare_s'] = _factor_time.monotonic() - start
    evidence['software'] = dict(torch=torch.__version__, cuda=torch.version.cuda, device=torch.cuda.get_device_name(dev))
    encoded = _factor_json.dumps(evidence, separators=(',', ':'))
    assert len(encoded) <= 13000
    print('FACTOR_Y_EVIDENCE ' + encoded, flush=True)
    torch.cuda.synchronize(dev)
    return state


def kernel_fn(data, state):
    m = data['hidden_states'].shape[0]
    if not state.get('_factor_enabled', False) or m <= MID_MAX_M or not state['sfx']:
        return _factor_original_kernel(data, state)
    return _factor_run(data, state, state['_factor_module'], state['_factor_scratch'])
