"""Bounded synthetic comparison for the repaired 9..16-token skinny backend."""
import json as _sk16_json
import time as _sk16_time

_sk16_original_prepare = prepare
_sk16_original_kernel = kernel_fn


def _sk16_synthetic(rows, pattern, state, seed):
    gen = torch.Generator(device='cpu').manual_seed(seed)
    route_rows = []
    for row in range(rows):
        if pattern == 0:
            local = torch.tensor([(2 * row) % 32, (2 * row + 1) % 32])
        elif pattern == 1:
            local = torch.cat((torch.tensor([0, 1]), torch.randperm(30, generator=gen)[:2] + 2))
        else:
            popularity = torch.arange(1, 33, dtype=torch.float32).pow(-1.25)
            local = torch.multinomial(popularity, 2, generator=gen)
        remote = torch.randperm(224, generator=gen)[:8 - len(local)] + 32
        ids = torch.cat((local, remote))[torch.randperm(8, generator=gen)]
        route_rows.append((ids + state['lo']) % 256)
    dev = state['w'][0].device
    x = torch.randn((rows, HIDDEN), generator=gen).bfloat16()
    if pattern == 2:
        x[-1] *= 2 ** -12
    ids = torch.stack(route_rows).to(device=dev, dtype=torch.int32).contiguous()
    tw = torch.rand((rows, 8), generator=gen).clamp_min(1e-6)
    tw = (tw / tw.sum(-1, keepdim=True)).to(dev)
    return x.to(dev), ids, tw


def _sk16_run(ext, state, x, ids, tw):
    out = torch.empty_like(x)
    ext.okc_run(state['w'], state['scratch'], x, ids, tw, out,
                state['lo'], state['cnt'], state['grid'])
    return out


def _sk16_probe(old, new, state, started):
    cases = []
    for rows in (9, 15, 16):
        for pattern in (0, 1, 2):
            if _sk16_time.monotonic() - started > 265:
                return {'ok': False, 'budget': True, 'cases': cases}
            x, ids, tw = _sk16_synthetic(rows, pattern, state, 20261029 + 31 * rows + pattern)
            expected = _sk16_run(old, state, x, ids, tw).float()
            got = _sk16_run(new, state, x, ids, tw).float()
            row_ref = torch.linalg.vector_norm(expected, dim=1)
            den = row_ref.clamp_min(row_ref.mean() * .1).clamp_min(1e-20)
            relative = (torch.linalg.vector_norm(got - expected) /
                        torch.linalg.vector_norm(expected).clamp_min(1e-20)).item()
            worst = (torch.linalg.vector_norm(got - expected, dim=1) / den).max().item()
            finite = bool(torch.isfinite(got).all().item())
            cases.append([rows, pattern, relative, worst, finite])
            if not finite or relative > .005 or worst > .02:
                return {'ok': False, 'cases': cases}
        # Analytic oracle: no local routes means exactly zero output. This also
        # exercises reuse of the shared global scratch between different modules.
        remote = ((torch.arange(8, device=x.device, dtype=torch.int32) + state['lo'] + 32) % 256)
        ids = remote.expand(rows, -1).contiguous()
        zero = _sk16_run(new, state, x, ids, torch.full_like(tw, .125))
        exact_zero = bool((zero == 0).all().item())
        cases.append([rows, 'remote', exact_zero])
        if not exact_zero:
            return {'ok': False, 'cases': cases}
    return {'ok': True, 'cases': cases}


def _sk16_compare(old, new, state, started):
    dev = state['w'][0].device
    stream = torch.cuda.Stream(device=dev)
    flush = torch.zeros((32768, 1024), dtype=torch.float32, device=dev)
    sink = torch.empty((1024,), dtype=torch.float32, device=dev)
    stream.wait_stream(torch.cuda.current_stream(dev))
    records = []
    with torch.cuda.stream(stream):
        for pattern in (0, 1, 2):
            if _sk16_time.monotonic() - started > 270:
                return {'ok': False, 'budget': True, 'patterns': records}
            x, ids, tw = _sk16_synthetic(16, pattern, state, 20261029 + pattern)
            arms = (old, new)
            outputs, graphs = [], [torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()]
            for arm in arms:
                _sk16_run(arm, state, x, ids, tw)
            stream.synchronize()
            for arm in (0, 1):
                with torch.cuda.graph(graphs[arm], stream=stream):
                    outputs.append(_sk16_run(arms[arm], state, x, ids, tw))
            events = []
            for repeat in range(8):
                for arm in ((0, 1) if repeat % 2 == 0 else (1, 0)):
                    torch.sum(flush, dim=0, out=sink)
                    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    begin.record(stream)
                    graphs[arm].replay()
                    end.record(stream)
                    events.append((repeat, arm, begin, end))
            stream.synchronize()
            pairs = {r: {} for r in range(8)}
            for repeat, arm, begin, end in events:
                pairs[repeat][arm] = begin.elapsed_time(end) * 1000
            ratios = sorted(pairs[r][1] / pairs[r][0] for r in range(8))
            ratio = (ratios[3] + ratios[4]) / 2
            records.append(dict(pattern=pattern, ratio=ratio,
                                previous_ns=[pairs[r][0] for r in range(8)],
                                candidate_ns=[pairs[r][1] for r in range(8)]))
    torch.cuda.current_stream(dev).wait_stream(stream)
    ratios = [r['ratio'] for r in records]
    gm = (ratios[0] * ratios[1] * ratios[2]) ** (1 / 3)
    return {'ok': gm < .97 and max(ratios) <= 1.02, 'geomean': gm, 'patterns': records}


def prepare(weights):
    started = _sk16_time.monotonic()
    state = _sk16_original_prepare(weights)
    state['skinny16_enabled'] = False
    evidence = {'selected': False}
    dev = state['w'][0].device
    if torch.cuda.get_device_capability(dev) == (9, 0) and _sk16_time.monotonic() - started <= 210:
        suffix = state.get('sfx', ())
        new = _ext(_SK16_DEFINES + suffix)
        old = _ext(tuple(MID_DEFINES) + suffix)
        probes = _sk16_probe(old, new, state, started)
        evidence['probes'] = probes
        if probes['ok']:
            pairs = _sk16_compare(old, new, state, started)
            evidence['paired'] = pairs
            state['skinny16_enabled'] = pairs['ok']
            evidence['selected'] = pairs['ok']
    else:
        evidence['skip_budget_or_device'] = True
    torch.cuda.synchronize(dev)
    evidence['preparation_elapsed_s'] = _sk16_time.monotonic() - started
    print('MOE_SKINNY16_EVIDENCE', _sk16_json.dumps(evidence, separators=(',', ':')), flush=True)
    print('MOE_SKINNY16_CHOICES', (16,) if state['skinny16_enabled'] else (), flush=True)
    return state


def kernel_fn(data, state):
    rows = data['hidden_states'].shape[0]
    if not state['skinny16_enabled'] or not 9 <= rows <= 16:
        return _sk16_original_kernel(data, state)
    cfg = data['config']
    assert cfg['local_start'] == state['lo'] and cfg['local_count'] == state['cnt']
    x = data['hidden_states'].contiguous()
    ids = data['topk_ids'].to(torch.int32).contiguous()
    if ids.data_ptr() & 15:
        ids = ids.clone()
    tw = data['topk_weights'].to(torch.float32).contiguous()
    return _sk16_run(_ext(_SK16_DEFINES + state.get('sfx', ())), state, x, ids, tw)
