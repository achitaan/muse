Continue the Open Kernel Challenge work in this repository. First read HANDOFF.md, GOAL.md, STATE.json, EXPERIMENTS.md and the official muse_bf16_proj/TASK.md and AGENTS.md.

The local session has handed off its work. You are the sole active controller for this checkout. Use only my existing Codex subscription and native Codex agents. Do not use paid GPU/cloud infrastructure, external model providers or API credits, and do not modify global orchestration settings.

Check the current time first. The original hard stop is October23,2026 at13:00 America/Toronto (17:00 UTC). If that deadline has passed, retrieve and summarize the already queued results but do not start a new optimization sprint or new evaluations until I give a new deadline.

The official H200 evaluator has received eight checks; two completed and six were queued at the handoff. Poll those exact job IDs from HANDOFF.md before creating anything new. No optimized speedup was verified at the snapshot: only the starter passed, at0.908671× SGLang. CPU models and SM90 compilation are useful evidence but are not H200 runtime validation.

Prioritize the measured router bottleneck. Compare the fused Triton hi/lo router, packed cuBLAS router, small-M GEMV/split-K candidates and adaptive partitioning. Integrate only measured correctness-passing improvements, verify the exact combined artifact, independently review it, and submit a legitimate improvement through the official CLI if time permits. Keep the best known correct file recoverable and record hashes, per-shape timings, confidence intervals, quotas and failures honestly.

Install the official okc CLI project-locally if needed. Authentication is not in Git: use the securely configured OKC_TOKEN environment secret or request the necessary official login once. Never print, commit or copy credentials into artifacts. A normal Codex Cloud environment may have no GPU; official H200 checks are the performance authority.

Continue useful work autonomously until the deadline, then stop all workers, preserve results and pending ideas, commit the handoff updates, and report the best verified speedup and submission status. Do not declare success from static checks or speculative performance claims.
