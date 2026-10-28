# MoE guarded FP32 SiLU experiment

Parent33114871 is fully checked5/6/integrity at2.365437416x, published2.348299935x.
Last daily MoE check reserved for this single bounded arithmetic hypothesis.
No baseline rerun, geometry sweep or rescore of the parent.

Current five SwiGLU sites use expf plus fast divide. The grouped Prefill source
instead uses SiLU(x)=x/2*(1+tanh(x/2)), one special-function instruction. Transfer
the identity to MoE, with the original exp/div for x<-4 to avoid cancellation
of1+tanh in the negative tail. All computations stayFP32 and finalSwiGLU
storage staysBF16. Paired Marlin roundings, all weights/layouts, WGMMA pipeline,
queue/barriers, FP32 norms, routing and kernel_fn stay unchanged.

Candidatef5737777 includes a prepare-only native diagnostic over all65280finite
BF16gates, generated independently of challenge data. Compare actual GPU helper
outputs to stableFP64SiLU, using max(abs(reference),1e-5) denominator. Require
finite values, scaled error<2e-4 and BF16product relativeL2<.001. The test also
reports the original exp/div error. Compilation and diagnostics share the
ordinary cached main module; no extra extension or weight copy is added.

CPU screen uses the same stableFP64reference, allfiniteBF16gates and12fixed
32x4096gate/up distributions (means-4/0/4,sigma.25/1/4/16), followed by signed
linear maps and row RMS. CPUscaled error7.994e-7; maximum normalized rowL2
.0003453. This is not a PTX emulator. NVIDIA's documented tanh.approx.f32
maximum relative error is2^-11, too loose to assert that CPUresults transfer
to hardware; the full-domain GPU diagnostic is mandatory.
https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-tanh

Performance is unmeasured. Negative-tail branching can add divergence and
cancel the saved SFU work; faster source-level math is not evidence of faster
whole-kernel execution. One fullcheck, fixed guard: all5tests/6benchmarks,
integrity, exact hash, numerical diagnostic, ≥1.5%whole-MoE gain over cached
check, no case<98.5%ofparent. At most one score, only if the new checked lower
confidence bound clears prior publication upper2.383524434 and the guard passes.
Do not transfer to Decode based on CPU or MoE alone; its whole-layer benefit
must be material. Phase2.45x is only a checkpoint; overall top50Goal stays active.

Closed22:52UTC: c6d1d5cf59b5441a passed5/6/integrity at2.348851666x, CI[2.313618891,2.384084441]. Native audit all65280gates finite; scaled error7.6682e-5 (original8.2428e-7), BF16stageL2.000143914. Actual precision audit passed, but aggregate regressed; controller discarded before guard. Five of six point ratios<1; soleM16gain.52%is noise, no selective retention. No score/Decode transfer/nearby cutoff sweep.
