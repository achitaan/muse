"""Prepare-only synthetic phase diagnostics. The timed incumbent stays exact."""
import base64 as _phase_b64
import bisect as _phase_bisect
import hashlib as _phase_hash
import json as _phase_json
import random as _phase_random
import statistics as _phase_stats
import struct as _phase_struct
import time as _phase_time
import zlib as _phase_zlib
from torch.utils.cpp_extension import load_inline as _phase_load

_phase_original_prepare = prepare
_phase_original_kernel = kernel_fn
_PHASE_QUERY = r'''
#include <c10/cuda/CUDAException.h>
std::vector<int64_t> okc_phase_info() {
  cudaFuncAttributes a{}, b{};
  C10_CUDA_CHECK(cudaFuncGetAttributes(&a, okc::moe_fused<false>));
  C10_CUDA_CHECK(cudaFuncGetAttributes(&b, okc::moe_fused<true>));
  int oa=0, ob=0;
  size_t sm=okc_smem_bytes();
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&oa, okc::moe_fused<false>, okc::NTL, sm));
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&ob, okc::moe_fused<true>, okc::NTL, sm));
  return {a.numRegs, b.numRegs, (int64_t)a.localSizeBytes, (int64_t)b.localSizeBytes,
          (int64_t)a.sharedSizeBytes, (int64_t)b.sharedSizeBytes, (int64_t)sm, oa, ob, okc::NTL};
}
'''


def _phase_module(state):
    defs = list(_main_defines(state['sfx'])) + list(state['sfx'])
    if DQ7 and '-DOKC_DQ7=1' not in defs:
        defs.append('-DOKC_DQ7=1')
    source = CUDA_SRC.replace('#define OKC_DEBUG 0', '#define OKC_DEBUG 1', 1) + _PHASE_QUERY
    tag = _phase_hash.sha256((source + repr(defs)).encode()).hexdigest()[:16]
    module = _phase_load(name='okc_moe_phase_' + tag, cpp_sources=CPP_SRC + '\nstd::vector<int64_t> okc_phase_info();\n',
                        cuda_sources=source, functions=['okc_run', 'okc_set_pdl', 'okc_phase_info'],
                        extra_cuda_cflags=['-O3', '-gencode=arch=compute_90a,code=sm_90a', '-lineinfo',
                                          '-DOKC_NS=3', '-DOKC_NSF=3', *defs], verbose=False)
    module.okc_set_pdl(0)
    return module


def _phase_input(m, dev):
    # Public-formula stress data, generated here from our own fixed seed.
    # These are not judge inputs and do not predict their route distribution.
    generator = torch.Generator(device=dev).manual_seed(20261029 + m)
    x = torch.randn((m, HIDDEN), generator=generator, device=dev).bfloat16()
    scores = torch.rand((m, 256), generator=generator, device=dev)
    z = torch.randn((256,), generator=generator, device=dev)
    u = torch.rand((256,), generator=generator, device=dev).clamp_min(1e-12)
    scores = scores + .03 * z / (-u.log()).sqrt()
    values, ids = scores.topk(8, dim=1)
    shard_counts = torch.bincount((ids // 32).flatten(), minlength=8)
    ids = ((ids - 32 * shard_counts.argmax()) % 256).to(torch.int32).contiguous()
    weights = (values / values.sum(1, keepdim=True)).float().contiguous()
    return dict(hidden_states=x, topk_ids=ids, topk_weights=weights,
                config=dict(local_start=0, local_count=32))


def _phase_metadata(data, grid):
    ids = data['topk_ids']
    local = (ids >= 0) & (ids < 32)
    counts = torch.bincount(ids[local].long(), minlength=32).cpu().tolist()
    jobs = [min(64, n - start) for n in counts if n > 8 for start in range(0, n, 64)]
    jobs += [n for n in counts if 0 < n <= 8]
    phases = []
    for costs, units in [([112, 135, 140, 158, 173, 185, 200, 207], 32),
                         ([100, 104, 115, 123, 132, 144, 156, 174], 16)]:
        cs = [costs[(n + 7) // 8 - 1] for n in jobs]
        prefix = [0]
        for cost in cs:
            prefix.append(prefix[-1] + cost * units * 16)
        boundaries = []
        for c in range(grid + 1):
            if not jobs or c == grid:
                boundaries.append(len(jobs) * units * 16)
            else:
                pos = prefix[-1] * c // grid
                job = _phase_bisect.bisect_right(prefix, pos) - 1
                boundaries.append(job * units * 16 + (pos - prefix[job]) // cs[job])
        phases.append(dict(split_units=len({x // 16 for x in boundaries[1:-1] if x % 16}),
                           stage_counts=[b - a for a, b in zip(boundaries, boundaries[1:])]))
    return dict(counts=counts, local_routes=sum(counts), active_experts=sum(n > 0 for n in counts),
                jobs=len(jobs), padded_rows=sum((n + 7) // 8 * 8 for n in jobs),
                zero_local_tokens=int((~local.any(1)).sum().item()), phases=phases)


def _phase_pack_trace(trace, indices):
    origins = [min(row[col] for row in trace) for col in range(8)]
    deltas = [trace[i][col] - origins[col] for i in indices for col in range(8)]
    assert all(0 <= x < 2 ** 32 for x in deltas)
    raw = _phase_struct.pack('<' + 'I' * len(deltas), *deltas)
    return dict(cta_indices=indices, column_origins_ns=origins,
                deltas_u32le_zlib_b64=_phase_b64.b64encode(_phase_zlib.compress(raw, 9)).decode(),
                full_grid=len(trace))


def _phase_trace_summary(trace, metadata):
    order = (0, 1, 7, 6, 2, 3, 4, 5)
    assert all(all(row[a] > 0 and row[a] <= row[b] for a, b in zip(order, order[1:])) for row in trace)
    base = min(row[0] for row in trace)
    intervals = dict(route=(0, 1), gather_and_range_wait=(1, 6), phase_A=(6, 2),
                     phase_B_including_readiness=(2, 3), post_B_to_grid_join=(3, 4), finalization=(4, 5))
    summary = {}
    for name, (a, b) in intervals.items():
        values = [row[b] - row[a] for row in trace]
        summary[name] = [min(values), _phase_stats.median(values), max(values)]
    latest_A = max(row[2] for row in trace)
    overlap = sum(row[3] < latest_A and metadata['phases'][1]['stage_counts'][i] > 0
                  for i, row in enumerate(trace))
    return dict(interval_min_median_max_ns=summary, trace_span_ns=max(row[5] for row in trace) - base,
                final_tail_after_latest_B_ns=max(row[5] for row in trace) - max(row[3] for row in trace),
                CTAs_finishing_nonempty_B_before_latest_A=overlap)


def _phase_case(m, state, module, scratch, flush, sink, stream):
    dev = state['w'][0].device
    data = _phase_input(m, dev)
    metadata = _phase_metadata(data, state['grid'])
    out = torch.empty((m, HIDDEN), dtype=torch.bfloat16, device=dev)
    outputs = [None, out]

    def call(arm):
        if arm == 0:
            outputs[0] = _phase_original_kernel(data, state)
        else:
            module.okc_run(state['w'], scratch, data['hidden_states'], data['topk_ids'],
                           data['topk_weights'], out, state['lo'], state['cnt'], state['grid'])

    stream.wait_stream(torch.cuda.current_stream(dev))
    with torch.cuda.stream(stream):
        graphs = [torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()]
        for arm in (0, 1):
            call(arm)
        stream.synchronize()
        for arm in (0, 1):
            with torch.cuda.graph(graphs[arm], stream=stream):
                call(arm)
        schedule = [(0, 1)] * 5 + [(1, 0)] * 5
        _phase_random.Random(20261029 + m).shuffle(schedule)
        events = []
        for repeat, order in enumerate(schedule):
            for arm in order:
                torch.sum(flush, dim=0, out=sink)
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record(stream)
                graphs[arm].replay()
                b.record(stream)
                events.append((repeat, arm, a, b))
        stream.synchronize()
        pairs = [{} for _ in schedule]
        for repeat, arm, a, b in events:
            pairs[repeat][arm] = a.elapsed_time(b) * 1_000_000
        reference, got = outputs[0].float(), outputs[1].float()
        error = torch.linalg.vector_norm(got - reference) / torch.linalg.vector_norm(reference).clamp_min(1e-12)
        den = torch.linalg.vector_norm(reference, dim=1)
        row_error = torch.linalg.vector_norm(got - reference, dim=1) / den.clamp_min(.1 * den.mean()).clamp_min(1e-12)
        relative, worst = float(error.item()), float(row_error.max().item())
        assert bool(torch.isfinite(got).all().item()) and relative <= .005 and worst <= .02
        trace = scratch[3].cpu().tolist()
    torch.cuda.current_stream(dev).wait_stream(stream)
    ratios = [p[1] / p[0] for p in pairs]
    record = dict(m=m, synthetic=True, seed=20261029 + m, metadata=metadata,
                  paired_original_ns=[p[0] for p in pairs], paired_instrumented_ns=[p[1] for p in pairs],
                  arm_orders=schedule, overhead_median_ratio=_phase_stats.median(ratios),
                  relative_to_incumbent=relative, worst_row_to_incumbent=worst,
                  timing_interpretation_allowed=.95 <= _phase_stats.median(ratios) <= 1.05,
                  summary=_phase_trace_summary(trace, metadata), trace=_phase_pack_trace(trace, list(range(len(trace)))))
    return record, trace


def prepare(weights):
    started = _phase_time.monotonic()
    state = _phase_original_prepare(weights)
    if (not state['sfx'] or torch.cuda.get_device_capability(weights['w13'].device) != (9, 0)
            or _phase_time.monotonic() - started > 185):
        print('MOE_PHASE_SKIP', 'scale eligibility or preparation budget', flush=True)
        return state
    module = _phase_module(state)
    if _phase_time.monotonic() - started > 270:
        print('MOE_PHASE_SKIP', 'compile exhausted diagnostic budget', flush=True)
        return state
    dev = weights['w13'].device
    scratch = _make_scratch(dev) + [torch.zeros((state['grid'], 8), dtype=torch.int64, device=dev)]
    flush = torch.zeros((32768, 1024), dtype=torch.float32, device=dev)
    sink = torch.empty((1024,), dtype=torch.float32, device=dev)
    stream = torch.cuda.Stream(device=dev)
    records, traces = [], []
    for m in (32, 128, 256):
        if _phase_time.monotonic() - started > 280:
            break
        record, trace = _phase_case(m, state, module, scratch, flush, sink, stream)
        records.append(record)
        traces.append(trace)
    resources = module.okc_phase_info()
    admission = resources[0:2] == [96, 96] and resources[2:4] == [0, 0] and resources[7:9] == [1, 1]
    for record in records:
        record['timing_interpretation_allowed'] = record['timing_interpretation_allowed'] and admission
    evidence = dict(synthetic_only=True, independent_jobs=1, repeats_are_not_independent=True,
                    incumbent_timed_source_unchanged=True, resources=resources,
                    resource_matches_historical_incumbent=admission, cases=records,
                    software=dict(torch=torch.__version__, cuda=torch.version.cuda,
                                  device=torch.cuda.get_device_name(dev),
                                  compute_capability=torch.cuda.get_device_capability(dev)))
    encoded = _phase_json.dumps(evidence, separators=(',', ':'))
    if len(encoded) > 12500:
        for record, trace in zip(records, traces):
            critical = sorted({i for col in (2, 3, 5) for i in sorted(
                range(len(trace)), key=lambda j: trace[j][col], reverse=True)[:16]})
            record['trace'] = _phase_pack_trace(trace, critical)
        evidence['trace_compaction'] = 'Union of latest16 A/B/final CTAs; summaries cover the full grid'
        encoded = _phase_json.dumps(evidence, separators=(',', ':'))
    assert len(encoded) <= 13000
    print('MOE_PHASE_EVIDENCE ' + encoded, flush=True)
    print('MOE_PHASE_PREPARE_S', _phase_time.monotonic() - started, flush=True)
    torch.cuda.synchronize(dev)
    return state
