# Identity-rescale trial rejected after control comparison

Check787962a762c24249, sourceed810c7a, passed all16tests/12benchmarks/integrity
at2.640210548x, interval[2.600607390,2.690219648]. This is real full-suite
correctness evidence for this exact source, including the new L1 flat kernels.
It is an off-board check, not a scored publication or a confirmed improvement.

Resources passed:128registers, zero local bytes, two resident blocks,
67584dynamic shared bytes,256threads and the unchanged128/56/200register
budgets. The actual new flat L1 profile cases are3/5. All three evaluator
baseline/reference/checker hashes match the fresh pinned identities.

| Evidence | Relative speedup versus exact cached9e922c35 |
| --- | ---: |
| Changed B16 full case | 1.038475556 |
| Changed B32 full case | 1.048031251 |
| Changed-case geometric mean | 1.043242463 |
| Ten unchanged controls, geometric mean | 1.037861777 |
| Changed/control ratio | 1.005184396 |
| Complete official aggregate ratio | 1.040681913 |

The aggregate rose about4.1%, but the new-path excess over controls is only
about0.52%, below the fixed4% requirement. Two unchanged sliding cases rose
about12.5%/13.2%; other controls moved differently. This job does not isolate
a material gain from removing identity rescaling. Cross-job samples are not
paired treatment/control jobs, and no significance claim follows from the
reported interval or these normalized point ratios. No retry just for noise.

The original guard first stopped at a formatting defect: the profile table
truncated false to fals..., while true fit. The source and all first logs
remain frozen. After archiving the discarded phase, the local parser was
changed to recognize only that exact unambiguous truncation. Scientific
thresholds, cases, resources and source were unchanged. The corrected parser
still rejects the result for insufficient affected/control gain. No GPU rerun
was used. CPU format tests preserve all13 original rejection cases and reject
three wrong/ambiguous spellings; an AST audit permits only this one assertion
change. See review-initial-parser.json, original-attention_skipscale_evidence.py,
format-parser-cpu-audit.json and format-corrected-verdict.json.

The controller reverted trial8a6b1a2 atf295051 and its history is archived at
autoresearch-results/archive/20261029-043111/. Retain9e922c35, published2.537000514x.
No score, best-source promotion or Decode transfer. The conditional Decode5af9
draft remains CPU-only and parked because its standalone gain condition failed.

One Attention check used. Derived from the fresh prelaunch snapshot minus this
job: Dense19,MoE16,Attention19,Decode20,Prefill20checks; ten scores per track,
94checks/50scores total. These are not a later live quota query. No new check
before05:22:50.375730UTC. The current window ends October29 at20:00Toronto;
quota remains a reserve. No job/controller remains active after this phase.

Reproduce terminal review with
`python experiments/prepared/review_attention_skipscale_result.py` and format
repair tests with `python experiments/prepared/test_attention_profile_format_cpu.py`.
The immutable raw result preserves all per-case observations and evaluator IDs.
