# Prefill exact-decode and paired-epilogue transfer

Retained parent93e57ab4: check9ca79c7f676f4b63, full5tests/8benchmarks/integrity,
1.639290758x; publication23e7c8b4c3e54869 at1.663469245x. Reuse cached baseline.
One candidate28d04382, one official check; no sweep or baseline rerun.

Hypothesis: the expert kernel occupies37..44%of the full layer in cached profiles.
Its original FP4 bit placement and scalar BF16 epilogues do more conversion work
than the already verified standalone MoE/Decode implementations. Transfer those
exact techniques while preserving the grouped Prefill pipeline and layout.
No CTA/register/tile scheduling changes. No activation/weight requantization.

convert8 now uses Lee Penkman's exact eight-operation FP4 placement, Apache2.0,
https://github.com/lee101/openkernels/blob/c565e373d34575c5f7df862b05f8be7619078b0f/muse_moe_w4a16/submission.py
Matching _pg__CODE_POS is a bijection of all32packed bits. Full Apache license
and attribution retained. BF16accumulator/global-scale/product roundings use
seven paired sites; SwiGLU and all post norms retain their FP32 intermediates.
GEMM2 stores the BF16 pair directly, avoiding an idempotent BF16→FP32→BF16 trip.

Builder's inverse ledger recovers every parent CUDA byte and other Python AST
except the paired helper/uses, decode expressions and code-bit position constant.
CPU audit:1,048,576codes/65,536scale bytes inverted, old/new fragment bits identical,
all256masked multiply operands exhausted,1,566,720BF16products matched independent
FP64RN-even oracle,131,072Prefill power-of-two encoding products unchanged.
These checks do not establish CUDA behavior or performance.

Fixed guard: full5/8/integrity; exact audited hash;≥1.5%whole-track gain over
cached check; no case below97%; expert profile gain relative to unchanged work
>2%geometric mean, none below95%. Profile normalization is a diagnostic with
different jobs/inputs, not a claim of causal statistical significance. Use one
scored run only if checked confidence interval clears the prior publication's
upper bound1.688421283, with no unresolved correctness/profile issue.

MMA32 remains disabled. Its skipping applies only to each expert's≤32row tail;
weight traffic/decoding unchanged and synchronous MMAs lose WGMMA overlap. No
runtime/occupancy evidence establishes a gain, so do not blindly enable it.
Pinned upstream tree has no TRICKS file; README offers no timings. Source's
three-buffer path documents15.5vs12.2us and compiler serialization; CIT combine
documents2.5..3x regression (MOE_CIT0). Neither is a new candidate.
Primary source: https://github.com/vjeux/okc-2026/blob/f35a41ee1c13e316dabbf772830d47b23b453393/muse_prefill_layer.py

The local1.75xphase target is an experiment checkpoint, not the overall Goal.
Prefill does not count toward overall placement; preserve that distinction.
