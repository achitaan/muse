# TR8 with the human-approved MoE boundary gate

Source `moe_tinyroute_bounded.py`,SHA25625e8efa38242ac524740bc9e1aa14419c0ad20ab02cd900ac83b4678ef6f5d8f,193796 bytes,is unsubmitted. It changes the prepare-time gate of the frozen TR8 source95833547. CUDA,extension builder,scratch allocator,all timed dispatch definitions and precision operations remain exact. The only other edit is the diagnostic label BITWISE to BOUNDED.

## Authorization and evidence

The human answered **Allow bounded MoE norm variance** to the pending numerical-limit question. The permission is restricted to MoE norm sums/final output. Decode and Attention retain their existing byte gates. No precision rule is waived.

Diagnostic9dc89929 measured75 fixtures at M1,3,8,16,32 with changing inputs,routes,scratch banks and repeats. Every compared inter/SwiGLU and FP32 ybuf matched bytes; variation began at norm sums and sometimes final BF16 output,including unchanged-parent controls. Recorded global/norm errors are within the newly approved global limits. The earlier diagnostic did not retain full per-row norms,so it does not establish the new worst-row gate. That must be evaluated on H200.

The earlier TR8 check selected no timed new path because every strict final-output byte gate failed. Reopening follows the explicit human gate change and localization evidence; it is not a repetition of a measured performance rejection or unchanged scored code. No new routing algorithm or norm-recompute implementation is introduced.

## Numerical gate

Every captured parent/candidate pair must have matching shapes/dtypes and finite valid fields. Inter/SwiGLU and FP32 ybuf must match bytes,including signed zero. Norm-sum relative L2 must be<=1e-5. Final output relative L2 must be<=0.001; the maximum row ratio must be<=0.005,using max(reference-row norm,0.1*mean reference-row norm,1e-12). All-nonlocal outputs remain exactly zero even with NaN hidden inputs.

The snapshot logic is copied from the diagnosed completed-generation bank observer. It checks generation increment and reads only the valid expert-major prefix. Host route counting and metrics occur only in prepare-time probes; the public timed path is unchanged. Both original banks and all clears/arrivals remain.

The original five route patterns,three changing seeds and M1..8 domain remain. The original performance protocol is retained: >3% improvement in both arm orders for both one-local and skewed populations. Unsupported/failed shapes remain parent. There is no parameter sweep or input-derived cached routing decision. M>8 paths and the LUT fallback remain original. All extensions must finish before graph capture; original preparation budget gates remain.

## CPU evidence and later admission

Source inverses prove the CUDA and every timed definition unchanged. Sixteen analytical gate cases cover signed zero,each norm/output threshold,official row floor,nonfinite fields,shape/dtype and empty valid prefix. Six completed-generation cases validate snapshot bank ownership. These tests establish the host gate,not CUDA correctness or speed.

No new NVRTC compile is needed for unchanged CUDA; prior TR8 PRMT/LUT resource evidence remains source-bound. Fresh H200 gates,all official tests/benchmarks/integrity,actual moe_tr8_fused execution and affected/control latency are required. Empty selection or no benchmark execution is not a new algorithm result. A check is not a scored submission; no publication follows automatically. Official reference correctness is still required: comparison against the retained parent does not replace it.

No Decode transfer is authorized by this variance permission. Its independent complete-layer exact-byte and latency gates remain mandatory.
