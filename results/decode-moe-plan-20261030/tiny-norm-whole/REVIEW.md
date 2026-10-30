# Whole-row tiny MoE norm hypothesis

This separate candidate starts from retained MoE33114871. It leaves routing unchanged and replaces the tiny WS phase-B norm statistics/finalizer. It does not combine TR8 routing with norm work. TR8 is closed on measured performance. The norm candidate has no H200 result yet.

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

## Resource guard revision and self-review, October30

Current payload is `690ef087234197c1b91c7cb99161dcceb56d3f2c671986613c90c716d9d05bf0` (337980bytes),CUDA `dcfd3b845c748711aefc27ff07f7ab7cbf7a16aed739fd5dc12f0e2a33a631ce`. The original unrun `2f725371` payload and its original evidence are archived in `pre-resource-2f725371/`.

Only the private CUDA/CPP host resource API and loader change. The entire old CUDA remains an exact prefix; the device prefix and all four rebuilt cubins are byte-identical. Parent definitions,prepare,launch,numerical gates and shared timing harness are unchanged. `resource-guard-rebinding.json` records this comparison rather than assigning old measurements to new code.

Before a new-path probe,the loader now queries actual CUDA function attributes and occupancy. It requires zero local/static shared bytes,at least one resident block/SM,214784dynamic shared bytes and288threads. The parent grid is the device SM count,so one block/SM covers its capacity. The guard is outside timed calls; seven mock native-boundary cases verify acceptance,rejection and cache reuse without a kernel call. Actual H200 values remain unobserved.

An independent FP64 RMSNorm/weighted-sum oracle passes480CPU fixtures across M1..8,five route masks,three seeds and four distributions,including wide values,zero rows and poisoned invalid storage. Maximum norm-relative error is7.159e-8; maximum BF16 output-relative error is7.233e-5 and worst-row error1.640e-4. Five deliberate reduction,denominator,masking,weighting and store defects are detected. This limited CPU model does not simulate GEMMs or CUDA FMA/rsqrt behavior and does not prove GPU byte or variance gates.

Compiled constexpr data gives Smem36576bytes,aligned base36608,phase-A110592,phase-B178176,dynamic total214784. NVIDIA's unmodified CUDA13.0 occupancy header,run on the CPU,predicts one block/SM for143/146registers,with215808allocated shared bytes including the driver reserve. Deliberately excessive register/shared inputs yield zero residency. Model capacities follow the [NVIDIA Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html);this is not a new device query. Local MinGW requires an unsandboxed process because sandbox startup hit a DLL-relocation error;no global installation or environment mutation was made.

Labeled self-review: precision and ownership requirements remain unchanged;all CPU/source checks pass;native host-API compilation,actual resources,replay correctness and speed still need the one focused check. Stop at any resource/numerical/strict3% gate failure. Empty choices or parent-only profiles are not a new gain. No Decode check or automatic scored publication follows.
