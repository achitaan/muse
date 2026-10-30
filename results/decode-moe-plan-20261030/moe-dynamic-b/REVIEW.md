# Dynamic assignment of existing B ranges: CPU rejected

The retained WGS kernel already has stage-level A/B readiness: xproducer waits for doneAs[j*16+stage]>=8. Generic phase-overlap proposals are not new work. The previous four-tile WGS,coarse range quanta,register redistribution,cluster multicast and phase diagnostics are retained as negative/limited history and are not rerun.

This prototype preserves the original logical K-stage partitions and fixup slots. After each physical CTA finishes A,it claims one existing logical B range through ctr[1]. Producers and consumers use that range; B fixup ownership remains logical rather than physical. The original math,weights,global scratch layouts,stage completeness and final reduction stay unchanged. No job-size or matrix-geometry sweep is introduced.

Both32-thread producer warps must stay alive through the new mid-CTA join. The first implementation adds one576-thread join and computes the range on consumer thread0. A targeted follow-up moves the calculation to the existing x producer,using a shared logical-ID field and two576-thread joins. Both give up the old early-B prefetch overlap; no net latency benefit is established.

The exact inverse recovers retained CUDA. CPU models cover288 logical-range assignments and720 role-arrival orders;5971 physical-owner substitutions would select the wrong fixup owner. These are bounded ownership models,not CUDA race validation or FP32 atomic-order equivalence. The two-join follow-up would need further lifetime review if it passed resource admission. Any later large-WGS numerical gate must inspect live xB using the corrected sampler.

Matched NVRTC screens for both BIG entries retain96registers and no C751 warnings. Parent stack/spill-store/spill-load fields are16/56/124 bytes. Both prototypes report80/284/532. A declaration-only forward adaptation is applied equally to parent and candidate for NVRTC; the initial missing declarations and error logs are preserved. Neither result measures H200 localSizeBytes or actual spill traffic.

The fixed no-resource-regression screen rejects both. First CUDA75f6ea4a; producer-range CUDA86b0200f. No GPU check,Python submission,score,Decode transfer or further register/geometry tuning is admitted. Source,models,compiler proxies and logs remain here. The independent tiny-norm candidate remains next.

A final bounded algebra correction removes a redundant first-stage owner lookup:the claimed logical range already owns that stage. Independent weighted-prefix models cover42cases and1456first segments,including empty ranges. CUDA`cbc5f4e2` still reports80/288/532stack/spill-store/spill-load bytes versus parent16/56/124. That lookup was not the spill cause. This structural correction also fails the fixed resource gate;no GPU job or further tuning is admitted. Evidence is in`producer-range/direct-owner/`.
