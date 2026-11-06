# Reproduce the serving preparation

Use one H200 and the exact image/commit in [PRECOMP_HANDOFF.md](PRECOMP_HANDOFF.md).
These commands consume GPU rental time but no official challenge quota. The completed
run is already saved; repeating it is optional. Keep the $50 session allowance and
stop the rental during substantial offline work. Do not enable the experimental MoE
acceptance set based on these operator tests.

The examples assume this repository is `/workspace/muse` and new SGLang checkouts
will be created at the paths shown. Existing unrelated checkouts should be preserved.

```bash
export PATH=/opt/sglang/bin:/usr/local/cuda/bin:$PATH
export MAX_JOBS=4
export TORCH_CUDA_ARCH_LIST=9.0a
git clone --no-checkout https://github.com/sgl-project/sglang.git /workspace/sglang-stock
git -C /workspace/sglang-stock checkout --detach a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f
cd /workspace/muse
mkdir -p /workspace/precomp-results

export PYTHONPATH=/workspace/sglang-stock/python
python serving/check_marlin_nvfp4_layout.py
timeout 10m python serving/check_marlin_nvfp4_layout.py --sglang-checkout /workspace/sglang-stock
timeout 10m python -u serving/bench_marlin_stock.py --sglang-checkout /workspace/sglang-stock \
  > /workspace/precomp-results/stock.jsonl

git -C /workspace/sglang-stock worktree add --detach /workspace/sglang-s1 a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f
git -C /workspace/sglang-s1 apply --index /workspace/muse/serving/s1_postnorm_v2.patch.diff
export PYTHONPATH=/workspace/sglang-s1/python
python -m pytest -q /workspace/sglang-s1/python/sglang/test/kernels/test_muse_spark_post_expert_norm.py
timeout 10m python -u serving/bench_precomp.py --sglang-checkout /workspace/sglang-s1 \
  --s1-patch serving/s1_postnorm_v2.patch.diff > /workspace/precomp-results/s1.jsonl
```

For the frozen experimental MoE, explicitly prepare/compile before graph capture:

```bash
python serving/build_native_source.py
timeout 10m python -u serving/check_native_marlin.py --stage tiles
timeout 2m python -u serving/check_native_marlin.py --stage gemm1 --tokens 128
timeout 2m python -u serving/check_native_marlin.py --stage full --tokens 128 256 32 \
  --patterns uniform hot remote --edge-scales
timeout 2m python -u serving/check_native_marlin.py --stage full --tokens 128 \
  --patterns uniform mixed --activation-scale 0.000001
timeout 2m python -u serving/bench_native_marlin.py > /workspace/precomp-results/moe.jsonl
timeout 2m python -u serving/check_native_replay.py
timeout 3m compute-sanitizer --tool memcheck --error-exitcode 3 \
  python -u serving/check_native_marlin.py --stage full --tokens 128 --patterns uniform remote

timeout 4m python -u serving/bench_native_marlin.py --correctness-only --blocks 1 \
  --tokens 0 1 8 16 24 31 32 33 40 48 56 63 64 65 80 96 112 127 128 144 160 176 192 208 224 240 255 256 257 \
  --patterns uniform mixed local hot remote padding --offsets 0 32 64 96 128 160 192 224
```

No numerical tolerance in these probes confers serving acceptance. They report each
stage, unchanged-stock controls, invalid-slot initialization and finite/zero guards.
The debug stage outputs are diagnostic allocations; complete-operation timings do
not include those extra debug stores. The persistent production workspace is shared,
and simultaneous calls using the same owner are unsupported.

The separate experimental patch can be rebuilt and tested without enabling it:

```bash
python serving/build_native_patch.py --sglang-checkout /workspace/sglang-stock
git -C /workspace/sglang-stock worktree add --detach /workspace/sglang-moe a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f
git -C /workspace/sglang-moe apply --index /workspace/muse/serving/moe_direct_marlin.experimental.patch.diff
export PYTHONPATH=/workspace/sglang-moe/python
python -m pytest -q serving/test_native_integration.py
git -C /workspace/sglang-moe apply --index /workspace/muse/serving/s1_postnorm_v2.patch.diff
python -m pytest -q serving/test_native_integration.py \
  /workspace/sglang-moe/python/sglang/test/kernels/test_muse_spark_post_expert_norm.py
git -C /workspace/sglang-moe diff HEAD --check
```

On a local checkout, regenerate the summary/patch-rule checks for the **saved** run:

```bash
python serving/summarize_precomp.py
```

Keep new measurements in their own directory rather than overwriting the preserved
run. Export sources, logs and checksums before stopping or deleting an ephemeral pod.
