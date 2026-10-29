"""Reconstruct retained track loads from saved evidence, without GPU/API work."""
import hashlib
import json
import math
from pathlib import Path
import re

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'STATE.json').exists())
MOE_SHA = '331148716e8d1ba92b45cd29725be997d44b3a5235af74d75770bb9444100d50'
DECODE_SHA = 'd9fe7558624a1f070bbacb493b94302e90ff9bf77cdd8d4239f9d7a83ab39988'
INPUTS = [
    ('moe', MOE_SHA, 'best/moe_verified.py',
     f'results/resumed-moe-phase4-20261028/{MOE_SHA}/profiles/submission.d600231c8a8244cb.json'),
    ('decode', DECODE_SHA, 'best/decoder_verified.py',
     'results/decoder-segment-repair-20261028/profiles/submission.9560c915e85f4f50.json'),
]


def gm(values):
    return math.exp(sum(math.log(x) for x in values) / len(values))


def profile_row(profile):
    # The saved official report rounds durations to 0.1us. It is not a timeline.
    rows, total = [], None
    for line in profile['report'].splitlines()[1:]:
        match = re.fullmatch(r'(.*?)\s+(\d+(?:\.\d+)?)\s+\S*s\s+(\d+(?:\.\d+)?)%\s+(\d+)', line)
        if match:
            rows.append(dict(kernel=match[1], us=float(match[2]), calls=int(match[4])))
        elif line.startswith('total'):
            total = float(line.split()[1])
    assert rows and total is not None
    moe = [x for x in rows if 'okc::moe_fused<' in x['kernel']]
    assert len(moe) == 1
    return dict(profile_sum_us=total, moe_us=moe[0]['us'],
                attention_kernel_us=sum(x['us'] for x in rows if '::attn_kernel<' in x['kernel']),
                reported_kernels=rows)


def main():
    studies = {}
    for track, sha, source_path, profile_path in INPUTS:
        source = (ROOT / source_path).read_bytes()
        assert hashlib.sha256(source).hexdigest() == sha
        raw = (ROOT / profile_path).read_bytes()
        saved = json.loads(raw)
        assert f'/{sha}/submission.py' in saved['submission']
        assert saved['summary']['passed'] is True and saved['summary']['status'] == 'ok'
        assert len(saved['benchmarks']) == len(saved['profiles']) == (6 if track == 'moe' else 12)
        profiles = {x['index']: x for x in saved['profiles']}
        loads = {}
        for bench in saved['benchmarks']:
            assert bench['status'] == 'pass'
            case = bench['case']
            assert case == profiles[bench['index']]['case']
            weight = case['weight']
            assert weight == (60 if track == 'moe' else 46 if case['layer'] == 'sliding' else 16)
            load = loads.setdefault(case['load'], dict(baseline_load_ns=0., retained_load_ns=0., cases=[]))
            load['baseline_load_ns'] += weight * bench['baseline_ns']
            load['retained_load_ns'] += weight * bench['median_ns']
            row = dict(index=bench['index'], weight=weight, tokens=case.get('bs', case.get('batch', 0) * case.get('q_len', 1)),
                       baseline_benchmark_us=bench['baseline_ns'] / 1000,
                       retained_benchmark_us=bench['median_ns'] / 1000,
                       layer=case.get('layer'), **profile_row(profiles[bench['index']]))
            assert math.isclose(bench['baseline_ns'] / bench['median_ns'], bench['speedup'], rel_tol=1e-12)
            load['cases'].append(row)
        assert len(loads) == 6
        for load in loads.values():
            load['load_speedup'] = load['baseline_load_ns'] / load['retained_load_ns']
        reconstructed = gm([load['load_speedup'] for load in loads.values()])
        assert math.isclose(reconstructed, saved['summary']['speedup'], rel_tol=1e-12)
        studies[track] = dict(source_sha256=sha, job=saved['job_id'],
                              profile_path=profile_path, profile_file_sha256=hashlib.sha256(raw).hexdigest(),
                              reconstructed_track_speedup=reconstructed, loads=loads)

    report = dict(cpu_only=True, new_gpu_jobs=0, official_overall_aggregation='unknown',
                  benchmark_times='Saved baseline_ns and median_ns fields; paired-normalized latencies, not profile durations',
                  profile_times='Rounded summed kernel durations from a separate profiler pass, not critical-path fractions',
                  studies=studies)
    folder = ROOT / 'results/retained-load-opportunities-20261029'
    folder.mkdir(exist_ok=True)
    output = folder / 'loads.json'
    data = (json.dumps(report, indent=2) + '\n').encode()
    if output.exists():
        assert output.read_bytes() == data, 'Frozen report collision'
    else:
        output.write_bytes(data)

    lines = ['# Retained-source load and profile screen', '',
             'Historical evidence only. No API call, new measurement or GPU job. Each load is equally weighted in its own track.', '',
             'Benchmark times below are the saved paired-normalized `median_ns` fields. Profile times are rounded summed kernel durations from a separate pass. They are not interchangeable, additive critical-path fractions or forecasts.', '',
             'Decode uses 46 sliding layers plus 16 full layers per load. MoE has one row per load with common weight 60. Reconstructed speedups exactly reproduce both saved track results; this does not establish overall-board aggregation. The geometric mean of these normalized load times is not asserted to equal the API `score_us` field.', '',
             'MoE source33114871 / jobd600231c8a8244cb: checked **2.365437416x**, not its published2.348299935x.', '',
             '| Load | Tokens | Benchmark us | Profile MoE us | Next evidence needed |',
             '| --- | ---: | ---: | ---: | --- |']
    for name, load in studies['moe']['loads'].items():
        row = load['cases'][0]
        question = 'Routing/completion exposure; preserve early weight prefetch' if row['tokens'] <= 4 else 'Distinct instruction/gather bottleneck; skinny16 is closed' if row['tokens'] <= 16 else 'Calibrated phase/tail trace before scheduling changes'
        lines.append(f"| {name} | {row['tokens']} | {row['retained_benchmark_us']:.2f} | {row['moe_us']:.1f} | {question} |")
    lines += ['', 'Decode sourceD9 / job9560c915e85f4f50: checked **1.990224880x**, unscored repair. S/F denotes sliding/full. Weighted time is `46*tS + 16*tF`, in milliseconds per 62-layer load.', '',
              '| Load | Tokens | Benchmark S/F us | Weighted load ms | MoE profile S/F us | Attention kernel S/F us | Profile sum S/F us |',
              '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name, load in studies['decode']['loads'].items():
        rows = {r['layer']: r for r in load['cases']}
        s, f = rows['sliding'], rows['full']
        assert s['tokens'] == f['tokens']
        lines.append(f"| {name} | {s['tokens']} | {s['retained_benchmark_us']:.2f}/{f['retained_benchmark_us']:.2f} | {load['retained_load_ns']/1e6:.3f} | {s['moe_us']:.1f}/{f['moe_us']:.1f} | {s['attention_kernel_us']:.1f}/{f['attention_kernel_us']:.1f} | {s['profile_sum_us']:.1f}/{f['profile_sum_us']:.1f} |")
    lines += ['', 'At M256 sliding, this exact retained D9 profile reports173.5us MoE and301.9us summed kernels. The earlier200.3/331.3us profile belongs to another historical source; do not assign it to D9. Variability between profiling and benchmark passes prevents interpreting their ratios as exposed fractions.', '',
              'The source already overlaps router/pooler and releases phase-A slices individually to phase B. The upcoming synthetic MoE diagnostic can identify candidates for investigation but cannot supply the missing whole-layer overlap timeline. Full attention has substantial kernel time at larger loads, while MoE spans all regimes; whole-layer gain still requires a matching validated integration.', '',
              'No accepted runtime hypothesis follows solely from this table. D1, skinny16, occupancy and cache/scale flag trials remain closed. Prefer the calibrated phase diagnostic before changing MoE ownership. Preserve all current sources and fallbacks.', '',
              'Reproduce: `python experiments/prepared/summarize_retained_loads.py`. Exact input hashes, row data and full reported kernel names are preserved in `loads.json`.', '']
    markdown = ('\n'.join(lines)).encode()
    target = folder / 'review.md'
    if target.exists():
        assert target.read_bytes() == markdown, 'Frozen review collision'
    else:
        target.write_bytes(markdown)
    print(json.dumps(dict(report=str(folder.relative_to(ROOT)), new_gpu_jobs=0,
                          reconstructed={k: v['reconstructed_track_speedup'] for k, v in studies.items()})))


if __name__ == '__main__':
    main()
