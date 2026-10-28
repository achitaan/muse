# Two-algorithm MoE selection

Baseline 33114871 passed full coverage at 2.365437416x; no new score was spent
because its interval overlapped the published source's check. Its 16/32-token
shapes remain weakest, around 2.00x and 1.64x respectively in the paired results.
Raw cross-job microsecond changes are not controlled performance comparisons.

For 9..16 tokens, the incumbent tiny split algorithm rereads weights when a
popular expert has more than eight tokens. For 17..32 tokens, the incumbent
WGS algorithm uses a wider producer/consumer grid. The existing mixed skinny/fat
fallback can cover each expert in one job at these token counts. It already has
all required packed layouts. This trial compares the incumbent with that one
alternative; no CUDA code, tile dimensions or numerical precision are changed.
One additional macro tuple may compile during prepare when PRMT scales are valid.

During prepare only, compare on three independent synthetic routing patterns:
diffuse, two consistently hot experts, and mixed popularity. All have unique
top-8 IDs per row, normalized positive weights and a busiest local shard. Seeds
are independent of official input records. Seven paired graph replays alternate
order, with a 128 MiB cold-L2 flush outside each measured interval. Require a
3% mean improvement and no pattern more than 5% slower, plus finite outputs and
relative L2 agreement within 0.005 to the incumbent. These repetitions measure
within-job variability; they are not independent official job replications.

Keep only metadata identifying the backend, never synthetic inputs, outputs or
graphs. The timed call chooses by shape bucket and performs the complete fresh
computation. Larger batches and up to eight tokens retain their exact dispatch.
Mixed fallback supports up to 64 rows per job, so at most 32 jobs are needed
for M<=32, within JMAX=88. Scratch routing capacity 3072 exceeds M*8<=256.
All scales, BF16 rounding boundaries, FP32 norm arithmetic and routes remain.

Use one full official MoE check against a cached baseline. Require full 5/6
coverage and a recorded nonempty backend selection before keeping an improved
point result. If no backend is selected, discard the selector even if the new
point estimate happens to be higher. Scored publication still requires the
conservative interval gate relative to the actually published source.

Publication follow-up: the selector chose no backend and was discarded. Its
full passing run independently repeated the simpler 33114871 timed CUDA. An
AST-folding proof and matching CUDA justified one scored confirmation of that
simpler candidate despite overlapping individual intervals. This revised the
earlier nonoverlap-only publication preference; it did not change correctness
requirements or count individual replays as independent jobs. See MOE_NOTES.md.
