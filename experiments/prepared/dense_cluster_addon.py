"""Two fixed large-token cluster backends; ordinary synthetic prepare selection."""
import time as _lg_time
import json as _lg_json
from torch.utils.cpp_extension import load_inline as _lg_load_inline

_lg_ext = None
_lg_original_prepare = _achita_wg_base_prepare
_lg_original_kernel = _achita_wg_base_kernel
_LG_CONFIGS = {'qkv_gate': (33, 128, 2304, 8192, 'gqx', 3),
               'pooler': (65, 128, 4096, 8192, 'gq', 2)}
_LG_COMPARE_ROWS = {'qkv_gate': (33, 64, 65, 96, 127, 128),
                    'pooler': (65, 96, 127, 128)}


def _lg_module():
    global _lg_ext
    if _lg_ext is None:
        _lg_ext = _lg_load_inline(
            name='okc_dense_cluster_' + hashlib.sha256(_LG_CUDA.encode()).hexdigest()[:16],
            cpp_sources=_LG_CPP, cuda_sources=_LG_CUDA, functions=['gq', 'gqx'],
            extra_cuda_cflags=['-O3', '-gencode=arch=compute_90a,code=sm_90a'],
            no_implicit_headers=True, verbose=False)
    return _lg_ext


def _lg_key(data):
    x, cfg = data['x'], data['config']
    layer = cfg['layer']
    if layer not in _LG_CONFIGS or x.ndim != 2 or x.dtype != torch.bfloat16 or not x.is_cuda:
        return None
    lo, hi, n, k, kind, cl = _LG_CONFIGS[layer]
    m = x.shape[0]
    if not (lo <= m <= hi and cfg['m'] == m and cfg['n'] == n and cfg['k'] == k and x.shape[1] == k):
        return None
    if not x.is_contiguous() or x.data_ptr() % 16:
        return None
    return layer


def _lg_run(x, plan):
    out = torch.empty((x.shape[0], plan['n']), dtype=torch.bfloat16, device=x.device)
    getattr(_lg_module(), plan['kind'])(plan['weight'], x, out, x.shape[0], plan['n'],
                                       plan['k'], 128, 1, plan['cluster'], 8)
    return out


def _lg_input(m, k, dev, seed, pattern):
    gen = torch.Generator(device=dev).manual_seed(seed)
    x = torch.randn((m, k), device=dev, generator=gen).bfloat16()
    if pattern:
        x[:, ::127] *= 8
        x[m - 2] *= 2 ** -12
    rows = torch.arange(0, m, 8, device=dev)
    x[rows] = 0
    cols = (rows * 137 + 19) % k
    x[rows, cols] = 2.0
    return x


def _lg_verify(plan):
    # These are self-generated activations and static weights, not timed data.
    rows = (33, 64, 65, 127, 128) if plan['kind'] == 'gqx' else (65, 96, 127, 128)
    cases = []
    for m in rows:
        for pattern in (0, 1):
            x = _lg_input(m, plan['k'], plan['weight'].device, 20261028 + m + 31 * pattern, pattern)
            got = _lg_run(x, plan)
            reference = x.double() @ plan['weight'].double().T
            relative = (torch.linalg.vector_norm(got.double() - reference) /
                        torch.linalg.vector_norm(reference).clamp_min(1e-20)).item()
            impulse = (torch.linalg.vector_norm(got[::8].double() - reference[::8]) /
                       torch.linalg.vector_norm(reference[::8]).clamp_min(1e-20)).item()
            finite = bool(torch.isfinite(got).all().item())
            row_ref = torch.linalg.vector_norm(reference, dim=1)
            row_den = row_ref.clamp_min(row_ref.mean() * .1).clamp_min(1e-20)
            row_error = torch.linalg.vector_norm(got.double() - reference, dim=1) / row_den
            worst = row_error.max().item()
            worst_impulse = row_error[::8].max().item()
            cases.append([m, pattern, relative, impulse, finite, worst, worst_impulse])
            print('DENSE_CLUSTER_ORACLE', dict(layer=plan['layer'], m=m, pattern=pattern,
                  relative_l2=relative, impulse_l2=impulse, finite=finite,
                  worst_row=worst, worst_impulse=worst_impulse), flush=True)
            if not finite or relative > .004 or impulse > .001 or worst > .02 or worst_impulse > .005:
                return {'ok': False, 'cases': cases}
    return {'ok': True, 'cases': cases}


def _lg_compare(plan, state, started):
    dev = plan['weight'].device
    flush = torch.zeros((32768, 1024), dtype=torch.float32, device=dev)
    sink = torch.empty((1024,), dtype=torch.float32, device=dev)
    stream = torch.cuda.Stream(device=dev)
    stream.wait_stream(torch.cuda.current_stream(dev))
    records = []
    with torch.cuda.stream(stream):
        for m in _LG_COMPARE_ROWS[plan['layer']]:
            for pattern in (0, 1):
                if _lg_time.monotonic() - started > 270:
                    return {'ok': False, 'budget': True, 'cases': records}
                x = _lg_input(m, plan['k'], dev, 20261028 + m + 31 * pattern, pattern)
                data = {'x': x, 'config': {'m': m, 'n': plan['n'], 'k': plan['k'], 'layer': plan['layer']}}

                def call(arm):
                    return _lg_original_kernel(data, state) if arm == 0 else _lg_run(x, plan)

                outputs = []
                graphs = [torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()]
                for arm in (0, 1):
                    call(arm)
                stream.synchronize()
                for arm in (0, 1):
                    with torch.cuda.graph(graphs[arm], stream=stream):
                        outputs.append(call(arm))
                events = []
                for repeat in range(7):
                    for arm in ((0, 1) if repeat % 2 == 0 else (1, 0)):
                        torch.sum(flush, dim=0, out=sink)
                        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        start.record(stream)
                        graphs[arm].replay()
                        end.record(stream)
                        events.append((repeat, arm, start, end))
                stream.synchronize()
                pairs = {r: {} for r in range(7)}
                for repeat, arm, start, end in events:
                    pairs[repeat][arm] = start.elapsed_time(end) * 1_000_000  # CUDA ms -> ns
                samples = [pairs[r][1] / pairs[r][0] for r in range(7)]
                ratio = sorted(samples)[3]
                records.append(dict(m=m, pattern=pattern, ratio=ratio,
                                    previous_ns=[pairs[r][0] for r in range(7)],
                                    candidate_ns=[pairs[r][1] for r in range(7)]))
                print('DENSE_CLUSTER_PAIR', dict(layer=plan['layer'], m=m, pattern=pattern,
                      ratio=ratio), flush=True)
    torch.cuda.current_stream(dev).wait_stream(stream)
    nearby_ok = all(c['ratio'] <= 1.02 for c in records)
    selected_ok = all(c['ratio'] < .97 for c in records if c['m'] == 128)
    return {'ok': nearby_ok and selected_ok, 'cases': records}


def prepare(weights):
    started = _lg_time.monotonic()
    state = _lg_original_prepare(weights)
    choices, evidence = {}, {}
    state['dense_cluster_choices'] = choices
    dev = weights['qkv_gate'].device
    if torch.cuda.get_device_capability(dev) != (9, 0) or _lg_time.monotonic() - started > 210:
        print('DENSE_CLUSTER_SKIP_BUDGET_OR_DEVICE', flush=True)
        print('DENSE_CLUSTER_CHOICES', (), flush=True)
        return state
    _lg_module()
    for layer, cfg in _LG_CONFIGS.items():
        if _lg_time.monotonic() - started > 255:
            print('DENSE_CLUSTER_SKIP_REMAINING_BUDGET', layer, flush=True)
            break
        lo, hi, n, k, kind, cl = cfg
        weight = weights[layer]
        if (weight.dtype != torch.bfloat16 or tuple(weight.shape) != (n, k) or
                not weight.is_contiguous() or weight.data_ptr() % 16):
            continue
        plan = {'weight': weight, 'layer': layer, 'kind': kind, 'cluster': cl, 'n': n, 'k': k}
        oracle = _lg_verify(plan)
        record = {'oracle': oracle}
        if oracle['ok']:
            paired = _lg_compare(plan, state, started)
            record['paired'] = paired
            if paired['ok']:
                choices[layer] = plan
        evidence[layer] = record
    # Compact complete evidence survives the service's bounded diagnostic tail.
    print('DENSE_CLUSTER_EVIDENCE', _lg_json.dumps(evidence, separators=(',', ':')), flush=True)
    print('DENSE_CLUSTER_CHOICES', tuple(sorted(choices)), flush=True)
    torch.cuda.synchronize(dev)
    return state


def kernel_fn(data, state):
    key = _lg_key(data)
    plan = state['dense_cluster_choices'].get(key)
    if plan is not None and plan['weight'].device == data['x'].device:
        return _lg_run(data['x'], plan)
    return _lg_original_kernel(data, state)
