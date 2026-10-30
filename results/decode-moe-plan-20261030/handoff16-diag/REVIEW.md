# Fixed M16/M32 handoff preflight

Source `experiments/decode_moe/dense_handoff16_diag.py`, SHA256 `b3560cdd3b91cf5a8324f842833e08a61de3e1f72cf64099c17193e91f240415`, 715,191 bytes. Complete retained Dense edc61a1e prefix and public kernel_fn are unchanged. Every added name is private except the prepare wrapper and the same hashlib import. The diagnostic returns the original state, enables no path and cannot establish a new Dense score result.

## Six fixed families

O16/O32 and U16/U32 are the exact already-prepared four-kernel bundle09b371d0, with an appended diagnostic producer. The complete bundle is ad4c2044. Norm/FPR16 uses the fixed depth5 path and Norm/FPR32 depth3. FPR kernel source f68f77c0 is an exact prefix; only a host resource query is appended (complete CUDA564d9452), with an isolated module name and extra export. The host builder reverses exactly to the retained version. Norm source52667230 is the existing scalar early-signal clone with HasRouter=false. Original weights,format/config/instance and fallback guards are retained.

FPR's last template boolean is FUSE=true. PDL is its runtime integer argument. Queries target the same fused kernel in all three arms; no unused non-fused specialization is presented as the parent. This was corrected during CPU review before GPU admission.

## Two kinds of evidence

Projection probes use a **synthetic optimistic ceiling**, not real Attention/MoE overlap. A graph node first copies fresh BF16 input. A single256-thread producer signals at entry, waits5000 nanoseconds and only then flips each row's first-column sign bit. The selected projection twin must wait before its activation reads. Parent and candidate have separate produced buffers and include them in byte comparisons. Other columns come from the preceding copy. Fixed-delay spin adds no precision change and creates no dependency on the consumer. Parent WG overrides are recorded; a shape with a different selected WG override is skipped rather than mislabeled as a pure PDL comparison.

The real norm/FPR chain has three fixed arms: original norm+PDL0,original norm+fixed incoming PDL,early norm+the same fixed incoming PDL. Byte probes compare BF16 norm/pooled outputs,FP32 residual and all256 FP32 logits across normal,zero and outlier inputs. The unused None router output is excluded. The early arm must beat BOTH the original parent and late-signal PDL control by >3% and >=1.5us in both arm orders; this rejects a result attributable only to the familiar incoming flag. No best-of-configuration choice is made.

The unchanged shared harness captures CUDA graphs,flushes256MiB L2 before each replay and uses25 samples/arm. Ordinary inputs are restored after adversarial byte probes. These medians are screening evidence,not joint confidence intervals. No output is cached from benchmark data.

## CPU proof and readiness

All component function ASTs match the frozen sources. Quad/FPR CUDA prefixes are exact. Norm call/precision checks and prior exact-source quad/FPR compiler/ordering records are retained. The only new device kernel,the producer,compiles with10 registers,no stack/spills and no shared memory. SASS signal32 precedes timer96,block barrier240 and first store560. Six ownership models confirm only one in-bounds first-column writer per row.

CPU protocol tests cover copied/fresh inputs,late sign publication,opposite orders,four norm/FPR output tensors,causal-floor rejection and terminal-report retention with a6000-character suffix. Independent module failures retain the other valid unit evidence; two mocked preparation cases also return the original state. No mocked timings are GPU measurements.

Four background builds perform no input-dependent work or kernel launch; all join before this diagnostic's probes. Actual host compilation,resources,graph behavior and <300s preparation remain H200 questions. Probes start only before274s and timing refuses new work after278s. Driver resources and exact plan guards close unsupported families. A partial/error/budget-stop report is not a successful complete validation. No blind retry or score follows.

## Admission and interpretation

One Dense check is proposed after fresh quota,task-identity,active-job and60-minute spacing admission. Dense has8 checks in the latest snapshot; this leaves7 and preserves the minimum4. Decode's last check and all scored attempts remain untouched. The benchmark path is the retained Dense source; a changed aggregate point has no new-algorithm meaning.

Historical full-layer3% raw-event budgets are about4.201/6.537us for M16 sliding/full and4.970/9.495us for M32. Synthetic component ceilings cannot simply be added as a forecast or establish the critical path. Any later Decode candidate still needs its exact integrated source,complete outputs/cache bytes,preparation headroom and >3% both-order whole-layer evidence before a13/13 check result or publication claim.

The current strict MoE and Decode numerical guards remain unchanged. The pending MoE variance question is not an exemption.

## Primary mechanism references

NVIDIA separates the scheduling signal from the dependency wait that establishes result visibility,and describes overlap as opportunistic. Correctness here permits fully serialized execution and keeps dependent reads after the wait. [CUDA Programmatic Dependent Launch](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html).

The PTX reference describes globaltimer in nanoseconds and warns that its behavior is target-specific. This diagnostic targets H200 and makes no cross-GPU timer portability claim. [PTX globaltimer](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-globaltimer).
