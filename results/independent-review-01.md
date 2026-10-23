# Independent review — 2026-10-23

Fresh native Astra/high reviewer inspected all three initial candidates and the official task/reference/checks. No blocking defect found for declared workload. This is code review and CPU arithmetic evidence, not GPU validation.

- Router CPU arithmetic model: 21 cases, three deterministic seeds, M=1,8,9,32,33,128,256, including outliers and probe rows. Largest global relative L2 2.591e-6; largest row error 3.164e-6; largest probe error 3.102e-6. All below official bounds.
- Negative control using BF16 router weights alone failed at 1.649e-3.
- Single-spike BF16 models exact in 15 cases.
- Grouped large-GEMM tile mapping bijective in 165 normal/tail combinations.
- No cached inputs, reduced BF16 weight precision, or harness access found.
- Conditional limitation: router kernel assumes prepared leading stride N=256, valid for declared router shapes but not arbitrary output slices. Generalization should pass weight stride separately.
- Recommended measured follow-up: compare split1/direct-output and split2 with split8 on wide-output small-M projections, which already have 128 N tiles.

Unresolved: Triton compilation, actual GPU arithmetic, graph capture and authoritative H200 timing.
