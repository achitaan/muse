# Retained-source load and profile screen

Historical evidence only. No API call, new measurement or GPU job. Each load is equally weighted in its own track.

Benchmark times below are the saved paired-normalized `median_ns` fields. Profile times are rounded summed kernel durations from a separate pass. They are not interchangeable, additive critical-path fractions or forecasts.

Decode uses 46 sliding layers plus 16 full layers per load. MoE has one row per load with common weight 60. Reconstructed speedups exactly reproduce both saved track results; this does not establish overall-board aggregation. The geometric mean of these normalized load times is not asserted to equal the API `score_us` field.

MoE source33114871 / jobd600231c8a8244cb: checked **2.365437416x**, not its published2.348299935x.

| Load | Tokens | Benchmark us | Profile MoE us | Next evidence needed |
| --- | ---: | ---: | ---: | --- |
| decode 1 | 1 | 24.85 | 30.8 | Routing/completion exposure; preserve early weight prefetch |
| decode 16 | 16 | 96.64 | 78.0 | Distinct instruction/gather bottleneck; skinny16 is closed |
| decode 32 | 32 | 125.17 | 114.7 | Calibrated phase/tail trace before scheduling changes |
| verify 1 | 8 | 62.71 | 87.3 | Distinct instruction/gather bottleneck; skinny16 is closed |
| verify 16 | 128 | 179.36 | 174.4 | Calibrated phase/tail trace before scheduling changes |
| verify 32 | 256 | 215.28 | 226.7 | Calibrated phase/tail trace before scheduling changes |

Decode sourceD9 / job9560c915e85f4f50: checked **1.990224880x**, unscored repair. S/F denotes sliding/full. Weighted time is `46*tS + 16*tF`, in milliseconds per 62-layer load.

| Load | Tokens | Benchmark S/F us | Weighted load ms | MoE profile S/F us | Attention kernel S/F us | Profile sum S/F us |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| decode 1 | 1 | 84.64/92.28 | 5.370 | 29.0/22.3 | 4.9/11.2 | 87.4/87.6 |
| decode 16 | 16 | 138.35/219.75 | 9.880 | 58.8/86.1 | 8.0/87.2 | 127.6/234.0 |
| decode 32 | 32 | 157.79/312.97 | 12.266 | 84.8/85.5 | 9.2/165.1 | 162.3/318.7 |
| verify 1 | 8 | 127.15/135.69 | 8.020 | 64.6/78.4 | 7.1/13.9 | 129.2/153.5 |
| verify 16 | 128 | 262.52/349.59 | 17.669 | 172.1/159.3 | 11.6/103.1 | 271.1/347.2 |
| verify 32 | 256 | 331.85/495.64 | 23.195 | 173.5/208.4 | 14.9/183.5 | 301.9/506.5 |

At M256 sliding, this exact retained D9 profile reports173.5us MoE and301.9us summed kernels. The earlier200.3/331.3us profile belongs to another historical source; do not assign it to D9. Variability between profiling and benchmark passes prevents interpreting their ratios as exposed fractions.

The source already overlaps router/pooler and releases phase-A slices individually to phase B. The upcoming synthetic MoE diagnostic can identify candidates for investigation but cannot supply the missing whole-layer overlap timeline. Full attention has substantial kernel time at larger loads, while MoE spans all regimes; whole-layer gain still requires a matching validated integration.

No accepted runtime hypothesis follows solely from this table. D1, skinny16, occupancy and cache/scale flag trials remain closed. Prefer the calibrated phase diagnostic before changing MoE ownership. Preserve all current sources and fallbacks.

Reproduce: `python experiments/prepared/summarize_retained_loads.py`. Exact input hashes, row data and full reported kernel names are preserved in `loads.json`.
