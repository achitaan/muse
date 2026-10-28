"""Bounded prepare-time comparison of existing MoE algorithms, not tile search."""
import time as _bkt_time


def _bkt_synthetic(rows, kind, state, seed):
    generator = torch.Generator(device='cpu').manual_seed(seed)
    ids = []
    for row in range(rows):
        if kind == 'diffuse':
            local = torch.tensor([(2 * row) % 32, (2 * row + 1) % 32])
        elif kind == 'hot':
            local = torch.cat((torch.tensor([0, 1]), torch.randperm(30, generator=generator)[:2] + 2))
        else:
            popularity = torch.arange(1, 33, dtype=torch.float32).pow(-1.25)
            local = torch.multinomial(popularity, 2, generator=generator)
        remote = torch.randperm(224, generator=generator)[:8 - local.numel()] + 32
        row_ids = torch.cat((local, remote))[torch.randperm(8, generator=generator)]
        ids.append((row_ids + state['lo']) % 256)
    device = state['w'][0].device
    x = torch.randn((rows, HIDDEN), generator=generator).to(torch.bfloat16).to(device)
    ids = torch.stack(ids).to(device=device, dtype=torch.int32).contiguous()
    weights = torch.rand((rows, 8), generator=generator, dtype=torch.float32).clamp_min(1e-6)
    weights = (weights / weights.sum(-1, keepdim=True)).to(device)
    return x, ids, weights


def _bkt_select(state, started):
    device = state['w'][0].device
    if torch.cuda.get_device_capability(device)[0] != 9 or _bkt_time.monotonic() - started > 250:
        return {}
    suffix = state.get('sfx', ())
    alternative = tuple(MAIN_FALLBACK_DEFINES) + suffix
    alternate_ext = _ext(alternative)
    choices = {}
    # Larger batches cannot use a tiny split backend: retain their incumbent.
    stream = torch.cuda.Stream(device=device)
    flush = torch.zeros(32 * 1024 * 1024, dtype=torch.float32, device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        for rows in (16, 32):
            incumbent = (tuple(MID_DEFINES) if rows <= MID_MAX_M else tuple(_main_defines(suffix))) + suffix
            if alternative == incumbent:
                continue
            incumbent_ext = _ext(incumbent)
            timings = []
            safe = True
            for pattern, kind in enumerate(('diffuse', 'hot', 'mixed')):
                if _bkt_time.monotonic() - started > 270:
                    safe = False
                    print('MOE_BACKEND_TIME_LIMIT', rows, flush=True)
                    break
                x, ids, weights = _bkt_synthetic(rows, kind, state, 20261028 + rows * 100 + pattern)
                outputs = [torch.empty_like(x), torch.empty_like(x)]
                extensions = [incumbent_ext, alternate_ext]

                def call(arm):
                    extensions[arm].okc_run(state['w'], state['scratch'], x, ids, weights,
                                           outputs[arm], state['lo'], state['cnt'], state['grid'])

                for _ in range(2):
                    call(0)
                    call(1)
                stream.synchronize()
                reference = outputs[0].float()
                relative = (torch.linalg.vector_norm(outputs[1].float() - reference) /
                            torch.linalg.vector_norm(reference).clamp_min(1e-20)).item()
                if not torch.isfinite(outputs[1]).all().item() or relative > 0.005:
                    safe = False
                    print('MOE_BACKEND_NUMERIC_REJECT', rows, kind, relative, flush=True)
                    break
                graphs = [torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()]
                for arm in range(2):
                    with torch.cuda.graph(graphs[arm], stream=stream):
                        call(arm)
                samples = [[], []]
                # Seven paired repetitions with alternating order and a 128 MiB
                # cold-L2 flush outside the measured interval for EACH arm.
                for repeat in range(7):
                    for arm in ((0, 1) if repeat % 2 == 0 else (1, 0)):
                        flush.add_(1)
                        begin = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        begin.record(stream)
                        graphs[arm].replay()
                        end.record(stream)
                        end.synchronize()
                        samples[arm].append(begin.elapsed_time(end) * 1000.0)
                medians = [sorted(sample)[len(sample) // 2] for sample in samples]
                timings.append(medians)
                print('MOE_BACKEND_PATTERN', rows, kind, medians[0], medians[1], relative, flush=True)
                del graphs, outputs, x, ids, weights, reference
            if safe and len(timings) == 3:
                ratio = sum(t[1] for t in timings) / sum(t[0] for t in timings)
                worst = max(t[1] / t[0] for t in timings)
                if ratio < 0.97 and worst <= 1.05:
                    choices[rows] = alternative
                print('MOE_BACKEND_DECISION', rows, ratio, worst,
                      'fallback' if rows in choices else 'incumbent', flush=True)
    torch.cuda.current_stream(device).wait_stream(stream)
    stream.synchronize()
    print('MOE_BACKEND_CHOICES', tuple(sorted(choices)), flush=True)
    return choices


_bkt_original_prepare = prepare


def prepare(weights):
    started = _bkt_time.monotonic()
    state = _bkt_original_prepare(weights)
    state['backend_by_bucket'] = _bkt_select(state, started)
    return state
