# Serving track log (opens 2026-11-10 09:00 PT, closes 2026-11-19 23:59 PT)
- Pinned SGLang a55ab7d fetched via `okc serve init` -> serving/sglang/{sglang,RULES.md,patch.diff}.
- Worktrees: wt-s1 (serve/s1-postnorm), wt-moe (serve/moe-port).
- Limits: 1 scored/day + 3 checks/day (checks = C=16 rung only, 35 min); scored = C 1/4/16/32 ladder on 8xH200, 90 min.
- Lever: port kernel-board kernels into SGLang (rules: "Faster kernels from the kernel problems plug in at the same places"). MoE first (our MoE 2.38x SGLang per call; ~185 KB fits 400 KB cap).
- Nov 5: agents: S1 post-norm PR patch (ses_e52dff1c1ffeqlkCcbIKXcdpST); MoE port design (ses_e52dfb137ffe4MlyiIHCwysql7).
- S1 patch done: serve/s1-postnorm cdabc2c, 7,375 B, 3 files (Triton cutoff override, Marlin small-batch route, test). PR text serving/S1_PR.md. Unmeasured at server level.
