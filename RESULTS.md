# Results

Two official H200 checks passed 9/9 tests and 30/30 scored shapes:

- Starter: 0.908671x SGLang, 95% interval [0.895041, 0.922301].
- Small-batch candidate: 0.912009x SGLang, 95% interval [0.898329, 0.925689].

Neither beats SGLang overall. The small difference between these runs is not an established improvement. Six optimization jobs remained queued at the final local snapshot. No scored submission was made.

See results/official-results.md and raw result JSON for per-shape evidence. CPU arithmetic and SM90 compilation do not establish H200 runtime performance.


## October 28 checkpoint

The current scored result is 1.521764418x (job 1183fb1c393c4422), rank 106 at 05:13 UTC. The best check is 1.522874509x (60a349bee6944644). All official tests and benchmark cases passed. The target top-50 placement is not achieved. Complete raw evidence and attributed sources are in results/sprint-20261028-evidence; phase1 and phase2 directories retain the controller history. See HANDOFF.md for the current deadline and evaluation quota.


## Final evaluation checkpoint

Best scored job 6131bc96427e4486: 1.527771327x, provisional rank 85, all 9 tests and 30 cases passed. Top-50 cutoff 1.5448x; target unmet. All 20 checks and 10 scored submissions are exhausted. Full scored-phase history, frozen sources and an independent metric audit are preserved under results/sprint-20261028-*. Further evaluation requires renewed quota and a new user deadline.
