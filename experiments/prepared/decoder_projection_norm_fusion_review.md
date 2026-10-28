# Decode projection/norm fusion: inspection packet only

Inspection closed without a GPU run: O1 uses1024threads,64outputrows/CTA,
128CTAs; O8 uses256threads,16rows/CTA,512CTAs. A last-block norm replaces
parallel row finalizers with work on one SM; a full-grid spin would require
unproven residency after adding registers. The saving ceiling on small-token
loads is marginal after fences and norm work. Preserve the two Decode checks
and move to the one-query FP8 QK Attention hypothesis. No fusion implemented.

No candidate or GPU evaluation exists. Retain corrected1fb5d05d, checked
1.988085030x/scored1.986659677x. Two Decode checks remain. MoE phase13 is
closed: actual two-CTA occupancy regressed every affected load; no transfer.
Main chat only, preserve the selected model and all BF16/FP32 boundaries.

Corrected check9a176eaead1a44ec profiles identify small-token opportunities:
M1 pooler14.5us/router11.8us on separate streams, O projection5.2us,
post-attention boundary4.9us, pre-expert norm/ID translation2.1us. M8 pooler
13.7us/router14.5us, O projection5.1us, post-attention boundary4.5us,
pre-expert norm/ID translation1.6us. These are summed profile times, not
critical-path measurements. Optimizing router alone is mostly hidden by pooling.

Inspect best/decoder_verified.py's _achita_dense_project, _achita_dense_kernel
and the actual selected backend before editing. M1 pooler is v18s, M8 pooler
smallm2; O8 is v18s. m16trace_ is O16, while the earlier14.6us attribution was
unpooling. Avoid repeating that mistake or unselected projection comparisons.

Potential coherent fusion paths:

- O projection plus post-attention norm/residual/pre-FFN norm. This removes a
  larger separate boundary than pre-expert norm alone. Read the pinned SGLang
  norm_residual CUDA implementation before writing formulas; do not infer its
  materializations from an FP32 pseudocode expression. The BF16 GEMM result,
  BF16 branch boundary, FP32 residual stream, BF16 gain and FP32 small-router
  input must match. Input stream remains a fresh input; outputs need their own
  storage and correct captured lifetimes.
- Pooler plus pre-expert norm, paired with routing-local-ID emission in the
  existing FP32 router's final block, could remove _pn_inputs_kernel entirely.
  Pooler and router retain their side-stream overlap, followed by the existing
  event wait. Avoid introducing a cross-stream spin dependency.

The exact current _pn_inputs_kernel contract is concrete: read BF16 pooler
output and BF16 pre_expert_norm._get_effective_weight(); FP32 sum(x*x)/4096,
eps=RMS_EPS; y=x*inv*gain in that order; store BF16. Its Triton launch has
num_warps4 and enable_fp_fusion=False. Preserve two separate FP32 products
where required. Local IDs are (global_id-local_start)&255, not a -1 mask;
the native MoE later rejects nonlocal values. If emitted by the router, retain
the original global topk_ids return value and eager record_stream ownership
for every new side-stream tensor consumed by the main stream.

Distributed projection CTAs cannot normalize a whole row until every output
tile is ready. Prove a last-block completion protocol with fenced writer stores,
well-defined scratch initialization/reset and repeated/alternating-shape reuse.
Do not insert a grid-wide spin barrier into a grid whose full residency is not
guaranteed. Include the cost of extra fences/barriers in the performance case;
removing a launch does not itself prove a gain. A small-token last CTA can
cover the whole4096/8192-column row, but its reduction/storage mapping must be
audited for every thread and row, including nearby M2..7.

Check an upper bound on the possible whole-layer gain before spending a run.
PN-only fusion saves at most1.6..2.1us on two of the six token bands, before
its replacement work. A broader projection/boundary fusion may be more useful.
Use own synthetic controlled CUDA-graph comparisons if needed, and retain every
sample. No actual-call preprocessing, cached results or input-dependent choice.

Preparation recently took253.3seconds against a300second limit. Prefer replacing
the selected module over compiling a second copy of every dense kernel. The
source is988827bytes against1MB; solve packaging before upload, using lossless
source compression/metadata pruning with exact decoded source checks. No score
until full13tests/12benchmarks/integrity and a source-bound, material timing gain.

If no credible benefit survives inspection, move to Attention rather than use
the two remaining checks on a low-ceiling or fallback-only experiment. Prefill
remains last and does not contribute to the overall board.
