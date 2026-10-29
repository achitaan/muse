# Decode: retire two unreachable router preparations

Source `df68deb60a132159d630928d19d93bcb9513de86b98a8c9517cfadd5e2e2f731`,
893,359 bytes, is frozen in `results/decoder-router-prepare-prune-20261029/`.
It derives from the CPU-only bc528949 package and remains GPU-unverified.

Only two dense preparation job arguments change from `weights` to `{}`:
`prepare_router` and `prepare_router2`. Both return `{'enabled': False}` before
CUDA work when no router weight is present. Their state keys stay available.
The pool still submits21 host jobs;19 now enter GPU preparation factories.

Every whole-layer call to `_achita_dense_project` names QKV, O, pooler or
unpooler. All reads of the two retired dense router states in runtime dispatch
are guarded by `layer == 'router'`. This condition is excluded by all five
public-entry projection calls, independently of token count, dtype, alignment
or whether another projection backend falls through. The original full-FP32
small router, fused pooler/router, wide router, hi/lo fallback and root router
weights remain intact. No numerical CUDA or timed function changes.

Restoring only those two arguments recovers the entire packaged-parent AST.
The parent package independently recovers all D9 source bytes and CUDA.
This proves the bounded source change; it does not provide a new whole-layer
GPU result or preparation-time measurement. Keep all verified/active sources.
No standalone GPU check or score is justified solely to retime this setup work.
Use it only as part of a justified later integration and validate that exact
whole-layer source before promotion.

Reproduce without GPU imports or submission execution:

```text
python experiments/prepared/build_decoder_router_prepare_prune.py
```
