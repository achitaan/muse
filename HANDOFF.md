# October 28 optimization handoff

The user changed the target from 2.5x to getting as close as possible to a top-50
placement by **2026-10-28 04:00 America/Toronto (08:00 UTC)**. Continue in the
main chat only, using the existing subscription and free official GPU service.

## Verified checkpoint

`best/verified.py` and the retained `candidates/active.py` have SHA256
`41a3c15f1f94560314f227adf718e21d4b17a03c5d58bc864e17827048c21bdf`.
Check `60a349bee6944644` passed all 9 tests, 30 benchmarks and integrity checks
at 1.522874509x, CI [1.500031391, 1.545717627]. Scored job
`1183fb1c393c4422` independently passed at 1.521764418x. The 05:13 UTC board
snapshot was rank 106; rank 50 was 1.541x. Placement is provisional and changes.

The implementation is attributed MIT-licensed Nathen Fernandes v18 at commit
7385e5d8f70980cfd5322ea254edc0b2b8ca90f7, with the additive decoder extended to
pooler/output/sliced-unpooler kernel families. Preserve its copyright and license.
The original public implementation was checked at 1.516234527x and scored at
1.509654730x. These differences are close to the judge's uncertainty floor.

## Current execution

The off-board check quota is exhausted (20/20); 2/10 scored submissions have
been used. Refresh `okc problems --json`. The next phase uses scored evaluations,
which still run official correctness before timing. Its baseline reuses the
already completed scored job above. Run the AutoResearch status command before
editing; do not manually commit/revert candidates or alter controller logs while
an experiment is active. The native Codex Goal remains the only controller.

The prepared next experiment adds 256-byte L2 prefetch hints. Its check attempt
was refused by quota before a GPU job launched and was automatically reverted.
The exact source remains in Git history and the unlaunched evidence directory.

## Findings

- The original retained packed-router baseline was 1.094835x; independent native
  CUDA work reached 1.243575x before adopting the stronger attributed implementation.
- Cold-cache setup tuning of existing routes, cuBLASLt algorithms, small-batch
  geometry, router compression and register limits did not improve the aggregate.
- Partial QKV groups initially caused a misaligned vector store. Masked scalar
  stores and aligned eight-weight groups fixed correctness but were slower.
- Applying the shorter exact decoder to QKV8 passed but did not improve the score.
- Coherent L2-only streamed weight loads passed but did not improve the score.

## Correctness and limitations

Read the official TASK.md and AGENTS.md. Compute every required product with
BF16 weights/activations and FP32 accumulation; the FP32 router may use the allowed
BF16 hi/lo split. All compression is reversible, with exact XOR escape restoration.
Fresh activations are processed each call; outputs are not cached. Preserve graph
capture and odd/outlier shape fallbacks. Extreme finite BF16 arithmetic can overflow
in the inherited implementation; this is not a general all-finite-input GEMM claim.

All raw completed check/scored results, frozen sources and controller records are
under `results/sprint-20261028-*`. Source hashes are local SHA256s, not server
content hashes. The October 23 handoff documents are historical snapshots under
`results/oct23-handoff-documents/`.

No credential is committed. Existing local WSL login is achitaan. In a fresh
Linux environment, configure OKC_TOKEN as a secret or complete official human
login. Check the clock before launching: no new evaluation after 08:00 UTC.
