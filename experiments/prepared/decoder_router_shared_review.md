# Next Decode hypothesis: one FP32 router input copy per CTA

Status: inspected design, not implemented or evaluated. Retain decoder09badc9b
(1.994319167x checked,1.993956388x published) and MoE33114871. All recent MoE
changes lost and are excluded. Main chat only; conserve evaluations.

The checked source's _RF_CUDA implements the full-FP32 small-token router in
route_fp32<TM>. Its128CTAs comprise16row tiles and8Ksplits; each256-thread CTA
has8warps, each computing two distinct expert rows. In every32-step Kloop,
all8warps load the same input elements for each token. The existing weights,
input and accumulators are FP32, with explicit __fmaf_rn, a fixed warp-shuffle
reduction, fixed8split reduction, sigmoid and deterministic biased top8/ties.

The cached profile09b34f4e921e49a6 reports route_fp32 at11.6us for one token
and14.9..17.9us for8tokens. Correct attribution: one-token pooler v18s_gemv is
~13.4us, unpooler smtrace_gemv~4.0us. Do not repeat previous projection/profile
mistakes or spend a run on pruning a small unpooler cost.

Candidate: __shared__ float xs[TM*1024], with every CTA thread cooperatively
loading the active M*1024input prefix once, then a full CTA barrier. Original
dot-product loads x[m*8192+ks*1024+j*32+lane] become
xs[m*1024+j*32+lane]. Keep every dot/FMA/reduction/routing operation and order
byte-for-byte otherwise. TM1uses4KB; TM8uses32KB (even when M2..7). All live
shared entries are overwritten from fresh input on every invocation, and
inactive m>=M entries are never read. No persistent activation/output cache.

Logical input global-load count per CTA falls from8192*M to1024*M:8x. Weight
loads remain16384floats/CTA,8MiB across the grid. This counts instructions and
logical traffic, not HBM bytes; original input loads can hitL1, and the new
copy/barrier can cost more than it saves. The gain is unmeasured. Shared reads
are bank-conflict-free for consecutive lane32indices. The128-block wave fits
132SMs with this modest shared footprint, but inspect actual resources and
side-stream co-residency before attributing a change in timing.

Implementation boundaries:

- Read/hash exact09badc9b and replace only the _RF_CUDA literal. Keep all other
  CUDA/Triton payloads and every other module AST node identical. No new module
  or weight bank; the existing router extension builds once during prepare.
- Rename the kernel to route_fp32_xshared in its definition/launch sites for
  observed profile evidence; leave the exported run signature unchanged.
- Preserve original128CTAs/256threads, counter increment128including wrap,
  volatile partials, fences, last-block logic, active warp masks and outputs.
- Add a barrier only before the dot loop, reached by every CTA thread. There
  must be no new barrier after an early return, nor private data/seed access.
- Keep FP32 router input/weights throughout; no BF16 hi/lo or TF32 substitute
  on this small path. Keep the larger fused pooler/router and routing intact.
- CPU-audit the actual shared index maps for M1..8, all8Ksplits, full8192input
  columns and every lane/warp. Use uint32payload patterns to prove copies
  preserve all float bits (including sign/subnormals), unique writers and no
  uninitialized active reads. Reverse the complete source patch and recover
  the original module AST. Do not manufacture expected numerical outputs.
- Require full13tests/12benchmarks/integrity and the existing FP32/norm guards.
  For this trial, require the new router kernel in profiles0/1/6/7, so an
  unselected fallback cannot be retained as a performance improvement.
- Use cached baseline1.994319167x. One focused full official check after review,
  no repeated GPU baseline. Inspect unchanged larger-token controls and the
  router spans. Score only a material verified improvement; no tile sweep.

The kernel source is ordinary native CUDA embedded in best/decoder_verified.py;
extract _RF_CUDA with ast.literal_eval for inspection. Existing resource APIs
and load_inline compilation are sufficient; no paid GPU, provider or subagent.
