# Per-CTA norm aggregation: CPU prototype only

This starts from retained33114871 and preserves its original finalizer. Unlike rejected690ef087,it adds no full-row ybuf scan and does not reduce finalizer parallelism. Phase-B unit squares/shuffles and all GEMM/SwiGLU/ybuf arithmetic stay original. Each CTA accumulates unit norm addends in shared job/token buckets,then publishes one nonzero partial per touched route after the existing helper join and before the original grid barrier.

Costs are explicit:2816additional shared bytes,704initial shared stores/CTA,one added full-CTA barrier before the producer/consumer fork,shared atomics,and a job-slot scan/publication. The static partition model predicts fewer global arrivals per route;it does not measure exposed atomic cost or latency. No speedup is claimed.

Global FP32 atomic addition flushes subnormal inputs/results;shared addition preserves them. The unit addend is explicitly flushed before shared aggregation to retain the original input behavior. Positive norm addends then cannot create a subnormal sum from normal inputs. Source:[NVIDIA PTX ISA](https://docs.nvidia.com/cuda/archive/12.8.0/pdf/ptx_isa_8.7.pdf). CPU72positive-addend cases have maximum relative regrouping change3.77896e-7 and detect the missing-FTZ mutant. This is a restricted scalar model,not GPU numerical validation.

CUDA7627e5a748216895447aa5a7a74dfb999aa418d10d9a406fb51ae0562a189143 compiles TINY/TINY2 PRMT at148registers with zero stack/spills. LUT builds,actual shared allocation/residency,complete helper/tail ownership and replay/lifetime proofs remain pending. An exact inverse before private device renaming recovers the parent. No Python payload exists;host API isolation still needs packaging.

Self-review:the original finalizer and global barrier remain;initialization precedes role fork and shared atomics;publication follows helper completion. Further independent read-only review is pending. No GPU admission,score,Decode transfer or incumbent change. A future check still needs exact-source binding,actual resource guards,all byte/approved variance gates and strict paired performance gates.


## Sparse revision after researcher review

The read-only researcher found no evident WS-tiny lifetime race in the initial prototype but flagged its avoidable barrier,initialization and dense scan costs. The initial source/evidence is archived in initial-dense/. The revised source initializes only intersecting B-job buckets inside the existing tid0 range-assignment block. That same thread reads its own range writes,then the existing WS_JOIN publishes initialization to helpers and consumers. Other consumers do not read newly written ranges before the join. Empty ranges are skipped. Publication scans only intersecting buckets after the final join. No extra barrier remains.

Current CUDA d8f90f0bdb7cbd9ca607e2990364ad0fcdb8e3d8a68874afe63dfaacdf621fa1 compiles all four TINY/TINY2 PRMT/LUT entries at148/145regs with zero stack/spills. Explicit static guards require tiny,WS,static scheduling and8-token jobs;the parent already traps fat jobs in this path. The scalar72-case audit passes again. This is parent self-review of the sparse revision;the researcher reviewed the earlier dense source and recommended sparse costs. H200 correctness,resources and profitability remain unverified.

Researcher ranked this bounded shared approach first for further packaging. A helper-private register accumulator is a distinct unimplemented alternative,with flush/register/lifetime risks;it has no performance evidence and is not GPU-admitted. No new GPU job or Decode check.


## Executable candidate packaged;still unrun

Payload e5c0e7ec4e28ccc2c9a91af85e41231a9ec356daaf0db13667c1560d6110b85b (338789bytes) preserves the parent definitions and shared harness. The checked host-resource bridge is reused with the group kernel namespace;only host API renaming/metadata are appended and the compiled device prefix remains byte-exact. Its guard requires actual zero local/static shared bytes,217600dynamic bytes,288threads and at least one block/SM before new-path probes. A comment-only prefix rename initially tripped the binding assertion;renaming is now restricted to the host portion.

Compiled constexpr layout:39392byte Smem,39424aligned base,217600total dynamic allocation. The offline occupancy calculator predicts one block/SM. Seven mock resource-boundary cases pass.396static ownership cases include2116empty CTA intervals;all processed buckets are initialized and exactly the touched valid buckets are published. The CPU model does not replace GPU byte,variance,replay or timing evidence.

Self-review:all inherited numerical/strict paired speed gates and all M>8 fallbacks remain unchanged. No private lazy compilation is reachable from timed dispatch. Actual H200 host compilation,resources,120private fixtures and full official tests/benchmarks/integrity remain pending. Admission is false;no new GPU job.
