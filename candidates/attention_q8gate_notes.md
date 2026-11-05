# q8 sliding gate scheduling

Check file: `candidates/attention_q8gate.py`.

The complete incumbent and phase helpers preceding the publication experiment
are unchanged bytes from `attention_pub.py` at 42a3d35. The unrelated L=1
publication experiment is not enabled.

## Source finding and one change

Both measured kernels already use cooperative per-CTA row slices, vectorized
partials, shuffle RMSNorm and vectorized BF16 output. v16S uses NT=256, while
the one-head v32S kernel uses NT=128. Widening rows would not increase v32S
parallelism, so that alternative was abandoned before a check.

The existing epilogue loads the gate only after all split publications finish,
then computes eight unchanged `sigm` values per participating thread. The
candidate publishes its original acq_rel arrival first, stages the gate and
sigmoids, then polls and executes the same barrier/merge/norm/output. An empty
volatile-asm register materialization prevents frontend sinking across the poll,
following the incumbent's existing nact/R guard. The last-arriver reset and epoch
update still happen immediately after its arrival. No additional CTA barrier.

Only cooperative L=8, sliding, NC1 rows fitting one thread wave use this path.
The other epilogues remain verbatim fallbacks. Runtime publication additionally
requires the exact scored input metadata and a successful per-shape gate.

## Gate

- Discover the actual incumbent route separately for v16S and v32S.
- Base/base read-flush ABBA control; exact output and all-cache bytes required.
- Reject a null median or paired ratio more than 1.5% from unity.
- Two independent candidate ABBA passes: median AND paired <=0.985 in EACH.
- Exact BF16 bytes and all KV cache bytes after poisoning new rows for each arm.
- No adoption until both complete; budget expiry preserves only completed gates.
- All timing gates precede optional nonfatal CPU metadata diagnostics.
- Every line begins `ATTN_Q8GATE` with shape, pass, base_us, cand_us,
  median, paired, null, exact and admitted fields.

## Local validation

`python3 scripts/test_attention_q8gate_cpu.py`: seven tests pass. Covers unchanged
prefix, generated CUDA anchors, identical normalization expression/order,
publication/gate/poll ordering, single-wave ownership, exhaustive arrival-order
models, two-pass/null/exact/budget rejection policy, and deferred diagnostics.
Python AST and forbidden-process scan pass. `git diff --check` passes.

No local Torch, CUDA, nvcc or GPU. GPU compilation, output byte equality and
performance require the caller's check. Register-pressure/codegen effects and
whether final SASS retains the intended overlap are unverified.
