# Decode boundary norm: source review

Read-only researcher review of retained c15df9d3 decoder_pdli.py and saved score452fbd9df6af4ff4. The attention boundary remains stock FusedPostNormResidualPreNorm at preparation6943 and layer calls6981/7863; PDL choices do not replace it. Saved source-bound profile times are4.8/4.7us atM1 sliding/full and4.7/4.7us atM8. These are historical measurements,not a current latency or recoverable-time estimate.

The retained H200 kernel already specializes hidden8192,oneCTA/row,256threads and four vec8 packs per thread. Rows remain in registers across both reductions. Gates/preweights are prefetched; scalar cluster size1 is used. Per-thread squares follow sequential fmaf; warp XOR masks16/8/4/2/1,then a shared-partial XOR tree over8warps pluszeros. Separate64byte shared scratch avoids a reuse barrier.

Postnorm has no postweight,epsilon1e-8,and a BF16 branch materialization before gated FP32 residual addition. Prenorm uses that FP32 residual,epsilon1e-5 and effective BF16 preweight. Output is BF16,while router input stores the unrounded FP32 normalized value. Do not replace it with BF16 output promoted back toFP32.

Gate logits are static weights,but coefficients are already prepared:decoder_pdli.py6925–6928 sets gate weights and calls precompute_coefficients;weights.attn_residual_gate enters at6946. The stock adapter reuses alpha/beta and contiguous FP32 copies. The timed norm kernel computes ga*stream+gb*normalized_branch and contains no gate-logit sigmoid. The attention-output activation sigmoid is a separate operation.

Sources:owned results/decode-moe-plan-20261030/pdl-norm/upstream/norm_residual.cuh71–267;decoder_pdli.py6925–6981/7863;profiles/decoder_pdli.452fbd9df6af4ff4.txt. Adapter:sibling opus-fresh/results/decoder-cluster-norm-reduction-diagnosis-20261029/pinned-upstream/fused_norm_residual.py323. No sibling files changed.

Decision:park generic single-CTA/fixed8192 and static-gate-precomputation proposals;both are already retained. Runtime-boolean pruning has no matched code-generation or exposed-time evidence and is not a GPU candidate. No payload,check or profiling job was created. This does not reopen closed DSM norms,router/Redux or _pn/_sfn folds.
