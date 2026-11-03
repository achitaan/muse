# Results

Two official H200 checks passed 9/9 tests and 30/30 scored shapes:

- Starter: 0.908671x SGLang, 95% interval [0.895041, 0.922301].
- Small-batch candidate: 0.912009x SGLang, 95% interval [0.898329, 0.925689].

Neither beats SGLang overall. The small difference between these runs is not an established improvement. Six optimization jobs remained queued at the final local snapshot. No scored submission was made.

See results/official-results.md and raw result JSON for per-shape evidence. CPU arithmetic and SM90 compilation do not establish H200 runtime performance.
