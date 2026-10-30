# Focused H200 admission: early norm / R2f handoff

Frozen source: `experiments/decode_moe/dense_norm_pair_diag.py`, SHA256 `67a4417e91158162f3e9db49fce4c7686ead9a0e1f7d3e5f9e2b04818be3d6eb`, 679,841 bytes. The complete retained Dense source edc61a1e is an exact prefix; its public kernel_fn AST remains unchanged. The diagnostic returns the original Dense state and enables no new timed path. Its Dense score point will not be treated as a new speed result.

## Question and exact bindings

Does moving the scalar norm's outgoing PDL signal before its incoming wait preserve every norm/router byte and reduce the real norm-to-R2f chain? The private norm source is exactly `f24c9d33e6be7f433a8c9629019f66f250b569f4cd235fd80b098499c399795e`, bound to the T1/T8 Decode draft 85a0ae65. R2f sources are the exact scored Decode c15 r1/r8 kernels, packaged into CUDA `e543374cbc0b93090f5dab714738abba8d966b48920fbeb15b60f1f4820afecf`. Both arms use the same incoming-PDL R2f; only the norm's outgoing signal changes. This is a causal test of a new predecessor signal, with no flag sweep.

The real original Dense pooler C12 plans and original FP32 router weights are read without modification. Prepare-only synthetic norm weights, channel/scalar gates, router bias and fresh branch/stream inputs define the probes. Normal, zero and outlier inputs change across three replays. Ordinary inputs are restored before timing. No benchmark input is inspected or cached.

## Filters and stopping conditions

- Current official baseline, reference, checker, TASK and AGENTS match the local pinned files with explicit UTF-8. Current template is retained but has no local pin.
- CPU ABI/router/source/dispatch audits for the private norm pass. Its original scalar arithmetic/reduction has an exact inverse and the HasRouter=true proxy has175 registers,64 bytes static shared,no spills.
- The exact two-namespace R2f device package compiles: both kernels64 registers,32768 bytes static shared,no stack/spills. Earlier exact-source ordering proofs and the scored c15 execution remain the prologue evidence.
- Mocked protocol audits cover changing/zero/outlier inputs, ordinary-data restoration and rejection of a1.4us gain despite a>3% ratio. A1.6us/>3% result must pass both arm orders. These are protocol tests, not GPU measurements.
- Original/private SDK modules and the R2f package compile in background without launching kernels. All three joins precede this diagnostic's captures. No installed file, module or environment is patched. Actual build contention and300s limit remain H200 questions.
- The norm and all six norm/router/R2f outputs must be finite and byte-identical. Both channel-wise and scalar gate metadata are checked at M1/M8. Timing uses the unchanged shared harness,25 cold-L2 graph samples per arm in both orders. A causal go result requires >3% and >=1.5us savings in both orders. It never activates a path.
- H200 capability and >=129 SMs are required. Driver norm/R2f resource checks reject local-memory use or an unlaunchable block. No same-SM co-residency is assumed: the norm is already resident and can finish independently, and the129-CTA R2f grid fits the132-SM H200 after it finishes.
- No new probes begin after274s; timing refuses work after278s. Compiler threads join rather than leak into timed calls. If preparation, identity, byte, resource or causal timing gates fail, retain the parent and do not spend the last Decode check on this candidate. No blind retry or automatic score follows.
- The compact terminal footer is bounded below6000 characters and retention-tested against the official16k log tail.

This one Dense check spends no Decode quota. A positive standalone result would still require matching Decode integration, whole-layer/cache bytes, preparation headroom and material weighted-load evidence before any scored publication. Existing MoE norm variance is not an exemption from byte gates.
