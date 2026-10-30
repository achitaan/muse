# Whole-row tiny MoE norm hypothesis

This separate candidate starts from retained MoE33114871. It leaves routing unchanged and replaces the tiny WS phase-B norm statistics/finalizer. It does not combine TR8 routing with norm work. TR8 remains first in the evaluation queue. Neither candidate has a new H200 result yet.

## Mechanism and cost

The source-bound ownership model finds256 norm atomic arrivals per valid local route:4096 output columns divided into16-column phase-B units. Exactly one helper-prefix or compute-tail warp owns each unit. The two WS epilogues retain their GEMM rounding, effective BF16 gain and FP32 ybuf stores; only squares,three shuffle reductions and the norm atomic are removed. Phase-A/SwiGLU,weight decoding,routing,queues,completion arrivals and bank clears remain original.

The finalizer transfers the retained MAIN two-pass row ownership to tiny NT256. One CTA owns each token,with16 columns/thread. Pass1 reads each valid4096-element FP32 ybuf row and reduces squares through eight warps and288 bytes of the freed shared ring. Pass2 reads that row again and applies the original epsilon1e-8,FP32 route weighting/summation and final BF16 store. Each local route has exactly one statistic writer; the completed bank remains observable by the approved gate. Nonlocal slots are masked before reads. No launch,allocation or grid barrier is added.

This doubles logical finalizer ybuf reads instead of repeating a full norm in four column CTAs. The extra requests are16KiB per local route,at most1MiB for M8/all-local. These are logical requests,not measured HBM traffic. Useful finalizer CTAs decrease from4*M to M; two CTA barriers and the second read may cost more than the removed atomics. There is no predicted speedup.

## Ownership and lifetime review

The unchanged WS helper issues only bounded units. Compute warps wait for each TMA-filled unit before consuming it; the helper prefix and compute-tail split covers each phase-B unit once. The existing WS_JOIN waits for every helper epilogue before the existing grid join. The new shared reduction starts after that join,when the weight ring is no longer consumed. The existing join already supports finalizer reads of every ybuf column; its arrivals,fences and generation update remain unchanged. GPU replay evidence is still required.

CPU models cover complete4096-column ownership,32 token-owner grid/shape cases and264 helper/tail partition cases. Private device namespace,host API names and extension cache isolate this module from the retained parent. The parent source remains an exact prefix; the private CUDA transformation has an exact inverse before symbol isolation.

## Resource screen and gates

Four NVRTC configurations cover TINY/TINY2 and PRMT/LUT. After removing a dynamic register-array index,all compile with zero stack frame and zero spills. PRMT uses146 registers; LUT uses143. Shared reduction storage reuses the existing ring. These are compiler results,not H200 residency measurements.

The payload reuses the human-approved MoE boundary gate: byte-exact inter/ybuf,finite fields,norm relativeL2<=1e-5,output relativeL2<=.001,worst-row<=.005 with the official-form denominator floor. Five route patterns and three changing seeds run for M1..8. Both one-local and skewed full-kernel graph comparisons must improve by more than3% in both arm orders,using the unchanged shared cold-L2 harness. New diagnostics stop starting at274s; timing stops at278s. Failed shapes retain the parent; all M>8 dispatch stays original.

No GPU job or publication is admitted by these CPU results. Fresh official correctness/integrity,new-path execution,resource/replay evidence and affected/control performance are still required. The MoE variance authorization does not change Decode or Attention byte gates.
