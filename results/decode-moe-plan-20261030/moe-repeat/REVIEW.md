# One MoE byte-failure localization check

Source `experiments/decode_moe/moe_repeat_diag.py`, SHA256
`2441f20829f1b7672ea8591983f2f9b8d952a0ae3e056d824926c88c00a478e9`,
240,378 bytes. This is a diagnostic, not a performance candidate. The complete
retained `33114871` prefix, its CUDA literal, `_ext`, public `kernel_fn` and
scratch factory remain unchanged. No new path can be selected or published.
Official benchmark timings from this check are parent timings, not evidence
that TR8 improved.

The known-failed TR8 CUDA (`b0ff69a1`) is observed through a separate source
constant, separate module cache and private launch function during prepare.
Its algorithm/flags are unchanged. This does not rerun a flag to seek a better
point; it obtains internal evidence absent from `9a4978c5acc24ed0`. Rejection
and strict byte gates remain intact regardless of the result.

## Question and fixed protocol

Did the earlier byte failure begin in routing/GEMM/SwiGLU, or do unchanged
intermediates feed order-sensitive normalization statistics? Does the parent
itself produce different bits on repeated identical inputs?

M1,3,8 cover the two changed tiny families and the exceptional earlier M3
failure; M16/M32 are unchanged MID/MAIN controls relevant to Decode transfer.
Each shape uses five fresh synthetic routing populations (none/one/four/all/
skew), three input seeds and two cycles. These are not official benchmark
draws. Nonlocal-only inputs contain NaNs and must still give exact zero.

Graphs A/B use the same parent state and distinct output buffers. C uses an
independent parent scratch allocation, sharing only immutable weights. TR8
uses C's scratch serially at <=8; at larger M, graph D is another parent.
Each cycle replays A,A,B,C,C,TR8-or-D. Consecutive same-graph replays separate
the output-buffer issue from generation/bank changes. Every generation and
bank is logged; comparisons must be interpreted with those values.

After each completed replay, capture valid expert-major prefixes of BF16
phase-A `inter`, FP32 `ybuf`, FP32 norm `ss`, and final BF16 output. The source
sets `S.ssp` using the launch-start generation and advances `ctr[3]` once at
its final global barrier. Hence the completed bank is `(ctr[3]-1)&1`.
The alternate bank starts at int32 offset 8+1024. Valid routes occupy the
unrounded expert-prefix range: `S.off[lane]=incl-tot`; only those rows are read.
No uninitialized padding/tails are compared. One extra scratch allocation
exists during prepare only; same-state launches are sequential.

Byte counts, finite checks, a few differing values, maximum absolute error and
relative L2 localize differences. Signed zero is a byte difference. No tolerance
is used to approve TR8. Identical inter/ybuf with differing ss is evidence
consistent with atomic sum ordering, not proof that the rejected route code
is correct for all inputs. A stable parent in finite trials is not a general
determinism proof. Do not relax the user's byte gates from this observation.

## Preflight and limits

CPU tests verify prefix preservation and unchanged public-function ASTs,
snapshot offsets for both generation banks, n=0/1/32/256, signed-zero and
nonfinite serialization, and 25 independently counted synthetic route
fixtures. The observed TR8 CUDA is byte-bound to the prior compiler evidence;
there is no new native algorithm to compile-screen. These tests do not validate
GPU synchronization or the snapshot interpretation on H200.

Current official TASK/AGENTS/baseline/reference/checker/NVFP4 sources match the
12:27 UTC pinned identities, including GPU=H200. Fresh quotas/jobs and the
60-minute spacing condition must be saved in admission before upload.
At most one new check, no scored run, no parameter sweep. Stop starting probes
at 278 seconds. Any instrumentation failure is reported separately; it must
not become a claim about TR8 arithmetic. Keep the final Decode check reserved.
