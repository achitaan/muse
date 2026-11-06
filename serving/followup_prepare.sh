#!/usr/bin/env bash
set -euo pipefail
export PATH=/opt/sglang/bin:/usr/local/cuda/bin:$PATH
export MAX_JOBS=4 TORCH_CUDA_ARCH_LIST=9.0a
mkdir -p /workspace/followup-results
cd /workspace/muse
git clone --filter=blob:none --no-checkout https://github.com/sgl-project/sglang.git /workspace/sglang-stock
git -C /workspace/sglang-stock checkout --detach a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f
git -C /workspace/sglang-stock worktree add --detach /workspace/sglang-s1 a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f
git -C /workspace/sglang-s1 apply --index /workspace/muse/serving/s1_postnorm_v2.patch.diff
export PYTHONPATH=/workspace/sglang-s1/python
python -c 'import torch,triton; print(torch.__version__,triton.__version__,torch.cuda.get_device_name())'
git -C /workspace/sglang-s1 rev-parse HEAD
nvidia-smi
timeout 12m python -u serving/check_s1_isolation.py --sglang-checkout /workspace/sglang-s1 \
  --s1-patch serving/s1_postnorm_v2.patch.diff --evidence-dir /workspace/followup-results/s1-inputs \
  > /workspace/followup-results/s1-isolation.jsonl 2> /workspace/followup-results/s1-isolation.stderr
echo S1_ISOLATION_FINISHED
