# CPU studies during the FP32-cache pacing wait

These files are separate from frozen trialf200e6e0. No GPU job or submission
was created for this study, and no tracked runtime source was changed.

The one-warp route model handles at most64entries (M<=8), using two32-entry
groups, match-any peers, a snapshot of first-group counts and a warp shuffle
for second-group offsets. The retained expert/job prefix and fat queue are
reused; original routing is the fallback for other paths. Eligible TINY/FAST1,
non-SPLIT, non-WGS routing uses one compute-CTA join rather than four. Weight
prefetch remains unchanged, and early_issue is a compile-time no-op there.
The full-mask shuffle stays warp-uniform; a warp barrier precedes the second
group's counter writes so every lane captures the first counts first.

The independent stable expert/index sort agrees with both the retained and
new metadata models in10611cases. Duplicate/nonlocal/empty inputs and the
32-entry boundary are included; three broken variants are rejected. This is
finite CPU evidence, not CUDA correctness, full-kernel progress or speed.

Both selected TINY and TINY2 compile in the NVRTC proxy for both BIG variants
at146registers and zero stack/spill fields, matching their retained counterparts.
TINY1 function text grows85248->105728bytes per variant. Added code can matter;
unchanged registers and fewer source barriers do not establish a runtime win.

Decision: park without a GPU check. Only MoE loadsM1 andM8 would change. Using
the source-bound cached paired-normalized times24.851390780/62.707423827us,
a hypothetical equal absolute saving must exceed1.548353529us from each load
to produce a1.5% MoE-track gain. Saving500ns from each would produce only
1.004732873x track improvement. These are dilution calculations, not forecasts.
No admissible routing critical-path measurement currently supports that bound.

The [lee101 public head](https://github.com/lee101/openkernels/commit/c565e373d34575c5f7df862b05f8be7619078b0f)
is still the pinned revision. The [vjeux MoE snapshot](https://github.com/vjeux/okc-2026/blob/f35a41ee1c13e316dabbf772830d47b23b453393/muse_moe_w4a16.py)
has exactly the retained general routing, WGS segment and phase text. It uses
the older decode placement and epilogues missing our retained Marlin BF16
boundaries/effective-gain rounding. It supplies no new mechanism to import.
The Apache2.0 license is saved alongside the source snapshot; text transport
added one final LF, documented in public-source-review.json. No public code
was imported into a submission.

Separate Decode handoff audit: Q/K normalization spans64columns per head,
but the selected M1 v18d QKV geometry is WR1/RT1,16columns per output cluster,
with K split across four CTAs. A partial tile cannot normalize independently.
It would need regrouped ownership or a fenced last-contributor head join,
charging extra atomics, scratch, preparation and altered projection resources.
AtM128 the retained QB geometry has NR64/G36/TM128/S3/PIPE539/RED5; its special
reduction distributes complete64-column heads in8-token groups across cluster
ranks, so a local head epilogue is more plausible. That source path only serves
M128; there is no measured overall opportunity yet. Neither path earns a Decode
GPU check from this source inspection. Keep D9 and all precision/fallback paths.

After the existing FP32-cache controller is terminal, preserve these CPU study
sources, models, compiler inputs/logs and this review in a results packet.
Do not edit tracked files outside its scope while that controller is running.
