"""Prepare-only controlled diagnostics; never touches actual call inputs."""
import json as _rr_json


def _rr_probe(state):
    rf = state['rf']
    if rf is None:
        print('ROUTER_REDUX_BITWISE', False, flush=True)
        return
    dev = rf['weight'].device
    stream = torch.cuda.Stream(device=dev)
    stream.wait_stream(torch.cuda.current_stream(dev))
    generator = torch.Generator(device=dev)
    generator.manual_seed(20261028)  # private generator; no global RNG mutation
    functions = (rf['ext'].run_incumbent, rf['ext'].run)
    with torch.cuda.stream(stream):
        zero_bias = torch.zeros_like(rf['bias'])
        for m in range(1, 9):
            for zero in (False, True):
                x = (torch.zeros((m, 8192), device=dev, dtype=torch.float32) if zero else
                     torch.randn((m, 8192), device=dev, dtype=torch.float32, generator=generator))
                bias = zero_bias if zero else rf['bias']
                outputs = [(torch.empty((m, 8), device=dev, dtype=torch.float32),
                            torch.empty((m, 8), device=dev, dtype=torch.int32)) for _ in range(2)]
                for arm in range(2):
                    functions[arm](x, rf['weight'], bias, rf['parts'], rf['counter'], *outputs[arm])
                assert torch.equal(outputs[0][0].view(torch.int32), outputs[1][0].view(torch.int32))
                assert torch.equal(outputs[0][1], outputs[1][1])
                if zero:
                    # Analytic: all logits0, sigmoid0=.5, zero bias; lowest IDs
                    # 0..7 win and eight equal raw scores normalize to1/8.
                    assert torch.equal(outputs[1][1], torch.arange(8, device=dev, dtype=torch.int32).expand(m, 8))
                    assert torch.equal(outputs[1][0], torch.full((m, 8), .125, device=dev, dtype=torch.float32))
        print('ROUTER_REDUX_BITWISE', True, flush=True)
        #128MiB exceeds H200 L2. This is owned diagnostic storage, not a view
        #into call data, and no synthetic outputs are returned as state.
        flush_buffer = torch.zeros(32 * 1024 * 1024, device=dev, dtype=torch.float32)
        for m in (1, 8):
            for pattern in ('normal', 'impulse'):
                if pattern == 'normal':
                    x = torch.randn((m, 8192), device=dev, dtype=torch.float32, generator=generator)
                else:
                    x = torch.zeros((m, 8192), device=dev, dtype=torch.float32)
                    x[:, 173] = 8192. ** .5  # same per-row RMS1 as normal inputs
                outputs = [(torch.empty((m, 8), device=dev, dtype=torch.float32),
                            torch.empty((m, 8), device=dev, dtype=torch.int32)) for _ in range(2)]
                def call(arm):
                    functions[arm](x, rf['weight'], rf['bias'], rf['parts'], rf['counter'], *outputs[arm])
                call(0)
                call(1)
                stream.synchronize()
                graphs = [torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()]
                for arm in range(2):
                    with torch.cuda.graph(graphs[arm], stream=stream):
                        call(arm)
                samples = [[], []]
                for repeat in range(8):
                    for arm in ((0, 1) if repeat % 2 == 0 else (1, 0)):
                        flush_buffer.add_(1)
                        begin = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        begin.record(stream)
                        graphs[arm].replay()
                        end.record(stream)
                        end.synchronize()
                        samples[arm].append(begin.elapsed_time(end) * 1_000_000.)
                ratios = [a / b for a, b in zip(*samples)]
                ordered = sorted(ratios)
                ratio = .5 * (ordered[3] + ordered[4])
                print('ROUTER_REDUX_AB', _rr_json.dumps({'m': m, 'pattern': pattern,
                    'incumbent_ns': samples[0], 'redux_ns': samples[1],
                    'paired_ratios': ratios, 'median_ratio': ratio}), flush=True)
                del graphs, outputs, x
    torch.cuda.current_stream(dev).wait_stream(stream)
    # Every call advances the counter by128; its low7bits still start at0.
    # Every consumed partial is overwritten by each future actual call.


_rr_original_prepare = prepare


def prepare(weights):
    state = _rr_original_prepare(weights)
    _rr_probe(state)
    return state
