> Latest focus: MoE experts (muse_moe_w4a16), at the user's request. Preserve quota. Read TRACK_NOTES.md for retrieval of attention/decoder results and pending jobs. Dot's additional ideas are pending.

> Resumed October 28: the user removed the deadline and requested top 50 overall, main chat only. Current focus is whole decoder layer, with conservative check-first use of quotas. Dense-projection quota remains exhausted. Attention preprocessing passed all 16 tests and 12 benchmarks at 1.120441x checked; scored job f6b899f674914ef2 is pending. The earlier sprint snapshots below are historical. See STATE.json.

# October 28 handoff

The unchanged user objective was to get as close as possible to top 50 on
muse_bf16_proj by 04:00 Toronto / 08:00 UTC, using the main chat and existing
subscription. The best provisional placement is **85**. Top 50 is unmet.

## Verified artifacts

`best/verified.py`: SHA256 `edc61a1e1fb26a10bcfacda3ef3fc06833c17ec7a078d5ded8eb8717509b765b`. Official scored job `6131bc96427e4486` passed all
9 tests, 30 benchmark cases and integrity checks at **1.527771327x**,
CI [1.504854757, 1.550687897]. The board cutoff for rank 50 was 1.5448x.
The aggregate score was independently recomputed exactly from the 30 raw case
records and six weighted load groups.

`best/lean.py`: SHA256 `c70c595b4f3ae1797f392f477266862fdd7246a4473f76d2f934177be3617d33`. Scored job `a91eb48178ad42be` passed the same
full validation at 1.525309711x. It removes an unused layout search and directly
compares the original versus retained unpooler32 layouts with FP64/probe guards
and cold-cache timing. The lean score and maximum score overlap within uncertainty.

## Execution is blocked

All 20/20 off-board checks and 10/10 scored submissions are used. There are no
remaining GPU evaluations on this problem today. The 00:00 UTC quota reset is
after the requested deadline. Do not start a new run until the user supplies a
new deadline and quota is available. No paid GPU service or external provider
is authorized. Check current time, quotas and board before resuming.

The native Goal is the sole controller. All work in this October 28 sprint was
performed in the main chat. The final experiment journal is archived in
`results/sprint-20261028-phase3`; do not fabricate missing events or manually edit
controller artifacts. `scripts/measure_scored.py` has the October 28 08:00 UTC
launch deadline. Changing it requires a new user deadline.

## Provenance and findings

The candidate derives from Nathen Fernandes' MIT-licensed v18 at commit
7385e5d8f70980cfd5322ea254edc0b2b8ca90f7. Its official baseline here was
1.516234527x checked and 1.509654730x scored. Local work extended its shorter
additive exact decoder to four kernel families, then evaluated a two-CTA
unpooler layout. Do not describe tiny aggregate point changes as statistically
proved improvements: one retained search selected no new paths.

Experiments also covered lossless compression, native CUDA MMA, cuBLASLt,
register limits, QKV grouping, L2 hints, aligned exponent windows and cluster
scheduling. Failed/slower attempts were retained in Git and reverted. A partial
QKV grouping exposed an alignment bug; scalar and aligned-vector corrections
passed but were slower. Two upload TLS failures were reconciled against server
jobs before retries; neither failed upload created an extra GPU job.

All BF16 values and allowed router hi/lo components retain their exact stored
precision. Products and partial sums accumulate in FP32. Fresh inputs are used;
outputs are not cached. The inherited implementation is not an all-finite-input
GEMM guarantee: extreme finite BF16 arithmetic can overflow. Preserve the odd,
outlier and nearby-shape fallbacks and all license notices.

Full source bytes, judge records, profiles and confidence intervals are in
`results/sprint-20261028-evidence`. The `unlaunched` directory records a refused
check launch; the same source was later evaluated in scored mode. Server content
hashes differ from local SHA256s and are not interchangeable.

No credential is committed. Existing local WSL authentication is achitaan. A
fresh environment needs an OKC_TOKEN secret or official human login. Never print
or commit the token. Read the official TASK.md and AGENTS.md before new work.
