# Conditional Decode improvement budget

This is arithmetic from retained D9 benchmark load times, not a speed forecast or an overall-board prediction. It uses no profiler duration as a critical-path fraction.

Assume only M32, M128 and M256 improve, by the same multiplicative load factor, with the other three loads unchanged. A desired track gain G then requires each changed load to speed up by G squared, removing `1 - 1/G^2` of its old weighted time.

| Desired Decode track gain | Each changed load time removed | M32 average layer saving us | M128 average layer saving us | M256 average layer saving us |
| ---: | ---: | ---: | ---: | ---: |
| 1% | 1.97% | 3.90 | 5.62 | 7.37 |
| 3% | 5.74% | 11.36 | 16.36 | 21.48 |
| 8% | 14.27% | 28.22 | 40.66 | 53.37 |

Average layer saving is `(46*delta_sliding + 16*delta_full)/62`. It does not require both layer types to improve equally. It charges every added operation, launch, store, reduction and lost overlap to the new complete layer time.

For a standalone MoE change affecting only M32, a1.5% track gain requires about9.34% speedup on that one load, or8.55% less load time. A3% per-load speedup on just one load gives only about0.49% track gain. These dilution identities constrain experiment value; they are not acceptance evidence.

Do not convert a synthetic phase span into any required saving above. A candidate must demonstrate the complete weighted layer change and unchanged controls, preserving whole-layer correctness. The diagnostic is intended to choose where to investigate, not to satisfy this performance gate.

Reproduce: `python experiments/prepared/screen_required_layer_savings.py`. Exact input hash and full-precision arithmetic are in `budget.json`.
