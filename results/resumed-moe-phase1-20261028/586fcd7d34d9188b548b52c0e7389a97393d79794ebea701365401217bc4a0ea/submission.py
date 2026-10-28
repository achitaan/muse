# okc: problem=muse_moe_w4a16
"""SGLang's NVFP4 expert path for Muse Spark Open on H200 (Marlin MoE runner), at SGLang commit
a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f: MuseSpark1xFp4Experts._forward_marlin, starting after the pre-expert norm.

prepare() repacks the experts into Marlin's layout with prepare_moe_nvfp4_layer_for_marlin,
as SGLang does at load. kernel_fn runs moe_align_block_size, two moe_wna16_marlin_gemm
(W4A16, FP32 reduce), silu_and_mul and the Triton post-expert-norm weighted sum.

Differs from SGLang: the expert all-reduce is not run."""

from typing import TypedDict

import types

import torch
from sgl_kernel.scalar_type import scalar_types

from sglang.kernels.ops.activation.activation import silu_and_mul
from sglang.kernels.ops.moe.moe_wna16_marlin import moe_wna16_marlin_gemm
from sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton import (
    fused_post_expert_norm_weighted_sum,
)
from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size
from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace
from sglang.srt.layers.quantization.marlin_utils_fp4 import prepare_moe_nvfp4_layer_for_marlin


class Weights(TypedDict):
    w13: torch.Tensor          # uint8 [256, 2I, H/2], gate rows [0, I) then up rows [I, 2I), packed E2M1
    w13_sf: torch.Tensor       # float8_e4m3fn [256, 2I, H/16]
    w13_gs: torch.Tensor       # float32 [256]
    w2: torch.Tensor           # uint8 [256, H, I/2]
    w2_sf: torch.Tensor        # float8_e4m3fn [256, H, I/16]
    w2_gs: torch.Tensor        # float32 [256]
    post_norm_w: torch.Tensor  # bfloat16 [H], the post-expert RMSNorm's weight (applied as 1 + w)


class Config(TypedDict):
    bs: int           # tokens, M
    d_hidden: int     # H = 4096, the width the MoE block runs its experts at
    d_expert: int     # I = 4096
    n_experts: int    # 256 experts in the model
    local_start: int  # this GPU computes experts [local_start, local_start + local_count)
    local_count: int  # 32: one GPU under 8-way expert parallelism
    top_k: int        # 8
    sharding: str     # "ep8"


class Inputs(TypedDict):
    hidden_states: torch.Tensor  # bfloat16 [M, H]
    topk_ids: torch.Tensor       # int32 [M, 8], global expert ids (0-255)
    topk_weights: torch.Tensor   # float32 [M, 8]
    config: Config


N_EXPERTS = 256
TOP_K = 8
# MuseSpark1xFp4Experts' candidates, smallest first.
BLOCK_SIZE_M_CANDIDATES = (8, 16, 32, 48, 64)
POST_NORM_EPS = 1e-8


def _marlin_layer(weights, start, count):
    """The parameters prepare_moe_nvfp4_layer_for_marlin reads, for experts [start, start+count)."""
    sl = slice(start, start + count)
    layer = types.SimpleNamespace(
        quant_config=types.SimpleNamespace(group_size=16),
        moe_runner_config=types.SimpleNamespace(is_gated=True),
        intermediate_size_per_partition=weights["w13"].shape[1] // 2,
        params_dtype=torch.bfloat16,
        w13_weight=weights["w13"][sl].contiguous(),
        w2_weight=weights["w2"][sl].contiguous(),
        w13_weight_scale=weights["w13_sf"][sl].contiguous(),
        w2_weight_scale=weights["w2_sf"][sl].contiguous(),
        # One global scale per expert for gate and up, as in SGLang's Marlin path.
        w13_weight_scale_2=weights["w13_gs"][sl].contiguous(),
        w2_weight_scale_2=weights["w2_gs"][sl].contiguous(),
    )
    prepare_moe_nvfp4_layer_for_marlin(layer)
    return layer


def prepare(weights):
    device = weights["w13"].device
    # One repacked bank: this GPU's 32 experts under EP8.
    banks = {}
    for start, count in ((0, 32),):
        layer = _marlin_layer(weights, start, count)
        expert_map = None
        if count != N_EXPERTS:
            expert_map = torch.full((N_EXPERTS,), -1, dtype=torch.int32, device=device)
            expert_map[start:start + count] = torch.arange(count, dtype=torch.int32, device=device)
        banks[(start, count)] = {
            "w13": layer.w13_weight.data, "w13_s": layer.w13_weight_scale.data, "w13_gs": layer.w13_weight_scale_2.data,
            "w2": layer.w2_weight.data, "w2_s": layer.w2_weight_scale.data, "w2_gs": layer.w2_weight_scale_2.data,
            "expert_map": expert_map, "count": count, "size_n": layer.intermediate_size_per_partition,
        }
    torch.cuda.synchronize()
    return {
        "banks": banks,
        "workspace": marlin_make_workspace(device, max_blocks_per_sm=4),
        # MuseSpark1xFp4Experts binds the effective (1 + w) weight of the post-expert norm.
        "norm_w": (weights["post_norm_w"].float() + 1.0).to(torch.bfloat16),
    }


def kernel_fn(data, state):
    cfg = data["config"]
    bank = state["banks"][(cfg["local_start"], cfg["local_count"])]
    x = data["hidden_states"].contiguous()
    num_tokens, size_k = x.shape
    size_n = bank["size_n"]
    gemm1_n = 2 * size_n
    expert_map = bank["expert_map"]

    local_ids = data["topk_ids"].to(torch.int32)
    if expert_map is not None:
        local_ids = expert_map[local_ids]
    local_ids = local_ids.contiguous()
    topk_weights = data["topk_weights"].to(torch.float32).contiguous()

    # With EP8 only local token/expert pairs reach the GEMMs. Use smaller
    # M tiles than the full-global-route-density heuristic for sparse groups.
    block_size_m = 8 if num_tokens <= 64 else 16
    if num_tokens == 1 and expert_map is None:
        from sglang.kernels.ops.moe.moe_align_single_token import moe_align_single_token

        sorted_ids, block_experts, n_padded = moe_align_single_token(local_ids, block_size_m)
    else:
        sorted_ids, block_experts, n_padded = moe_align_block_size(
            local_ids, block_size_m, N_EXPERTS if expert_map is not None else bank["count"])

    cache2 = torch.empty((num_tokens * TOP_K, size_n), device=x.device, dtype=x.dtype)
    cache13 = torch.empty((num_tokens * TOP_K * max(gemm1_n, size_k),), device=x.device, dtype=x.dtype)
    cache1 = cache13[: num_tokens * TOP_K * gemm1_n].view(-1, gemm1_n)
    cache3 = cache13[: num_tokens * TOP_K * size_k].view(-1, size_k)
    common = dict(moe_block_size=block_size_m, mul_topk_weights=False, is_ep=expert_map is not None,
                  b_q_type=scalar_types.float4_e2m1f, is_k_full=True, use_atomic_add=False,
                  use_fp32_reduce=True, is_zp_float=False)
    ws = state["workspace"]
    cache1 = moe_wna16_marlin_gemm(
        x, cache1, bank["w13"], None, bank["w13_s"], bank["w13_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, topk_weights, top_k=TOP_K,
        size_m=num_tokens, size_n=gemm1_n, size_k=size_k, **common)
    silu_and_mul(cache1.view(-1, gemm1_n), cache2, expert_ids=local_ids.view(-1), expert_step=1)
    y = moe_wna16_marlin_gemm(
        cache2, cache3, bank["w2"], None, bank["w2_s"], bank["w2_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, topk_weights, top_k=1,
        size_m=num_tokens * TOP_K, size_n=size_k, size_k=size_n, **common).view(num_tokens, TOP_K, size_k)
    return fused_post_expert_norm_weighted_sum(
        y, topk_weights, norm_weight=state["norm_w"], eps=POST_NORM_EPS, weight_first=True,
        slot_expert_ids=local_ids)

# Fuse the small-batch per-expert post norm and weighted sum. Larger batches
# retain the already-fused SGLang implementation.
import triton
import triton.language as tl


@triton.jit
def _achita_moe_small_finalize(Y, WEIGHT, NORM, IDS, OUT,
                              H: tl.constexpr, K: tl.constexpr,
                              EPS: tl.constexpr, BK: tl.constexpr):
    token = tl.program_id(0)
    expert = tl.arange(0, BK)
    dim = tl.arange(0, H)
    ids = tl.load(IDS + token * K + expert, mask=expert < K, other=-1)
    active = (expert < K) & (ids >= 0)
    values = tl.load(Y + (token * K + expert[:, None]) * H + dim[None, :],
                     mask=active[:, None], other=0).to(tl.float32)
    norm = tl.load(NORM + dim).to(tl.float32)
    # SGLang applies the effective BF16 norm weight before normalizing.
    scaled = (values * norm[None, :]).to(tl.bfloat16).to(tl.float32)
    inv = tl.rsqrt(tl.sum(scaled * scaled, 1) / H + EPS)
    weights = tl.load(WEIGHT + token * K + expert, mask=active, other=0)
    result = tl.sum(scaled * (inv * weights)[:, None], 0)
    tl.store(OUT + token * H + dim, result)


_achita_original_finalize = fused_post_expert_norm_weighted_sum


def fused_post_expert_norm_weighted_sum(y, topk_weights, norm_weight=None,
                                        eps=1e-8, weight_first=False,
                                        slot_expert_ids=None):
    if (y.shape[0] <= 16 and y.shape[2] == 4096 and y.shape[1] == 8
            and weight_first and norm_weight is not None and slot_expert_ids is not None):
        result = torch.empty((y.shape[0], y.shape[2]), device=y.device, dtype=y.dtype)
        _achita_moe_small_finalize[(y.shape[0],)](
            y, topk_weights, norm_weight, slot_expert_ids, result,
            y.shape[2], y.shape[1], eps, 8, num_warps=8)
        return result
    return _achita_original_finalize(y, topk_weights, norm_weight=norm_weight,
                                     eps=eps, weight_first=weight_first,
                                     slot_expert_ids=slot_expert_ids)

# Direct original-layout NVFP4 GEMV for a single token. All other shapes retain
# the checked Marlin path. No weights are expanded or rounded in prepare.
from torch.utils.cpp_extension import load_inline
import hashlib

_ACHITA_FP4_CPP = r'''
#include <torch/extension.h>
void achita_gate_up(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t);
void achita_down(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t);
'''

_ACHITA_FP4_CUDA = r'''
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>

// E4M3FN includes finite exponent-15 values, subnormals and signed zero.
__device__ __forceinline__ float fp8_scale(unsigned char bits) {
  unsigned e=(bits>>3)&15, m=bits&7;
  float value;
  if(e==0)value=float(m)*(1.0f/512.0f);
  else if(e==15 && m==7)value=__uint_as_float(0x7fc00000u);
  else value=__uint_as_float(((e+120u)<<23)|(m<<20));
  return (bits&128)?-value:value;
}
__device__ __forceinline__ float fp4_value(unsigned code) {
  float magnitude=float((0xc8643210u>>(4*(code&7)))&15u)*0.5f;
  return (code&8)?-magnitude:magnitude;
}
__device__ __forceinline__ float warp_sum(float value) {
  #pragma unroll
  for(int step=16;step;step>>=1)value+=__shfl_down_sync(0xffffffffu,value,step);
  return value;
}

template<int WARPS=8,int ROWS=4>
__global__ void fp4_gate_up_kernel(const __nv_bfloat16* __restrict__ x,
    const uint32_t* __restrict__ packed,const unsigned char* __restrict__ sf,
    const float* __restrict__ gs,const int* __restrict__ ids,
    __nv_bfloat16* __restrict__ out,int start) {
  constexpr int K=4096,I=4096;
  int slot=blockIdx.y,expert=ids[slot]-start;
  if(expert<0 || expert>=32)return;
  int lane=threadIdx.x&31,warp=threadIdx.x>>5;
  int row0=(blockIdx.x*WARPS+warp)*ROWS;
  float gate[ROWS]={},up[ROWS]={};
  #pragma unroll 1
  for(int kk=0;kk<K;kk+=256) {
    __nv_bfloat16 input[8];
    #pragma unroll
    for(int j=0;j<8;j++)input[j]=x[kk+lane*8+j];
    #pragma unroll
    for(int rr=0;rr<ROWS;rr++) {
      int row=row0+rr;
      uint32_t g=packed[((long)expert*(2*I)+row)*(K/8)+kk/8+lane];
      uint32_t u=packed[((long)expert*(2*I)+I+row)*(K/8)+kk/8+lane];
      float gsf=fp8_scale(sf[((long)expert*(2*I)+row)*(K/16)+kk/16+lane/2]);
      float usf=fp8_scale(sf[((long)expert*(2*I)+I+row)*(K/16)+kk/16+lane/2]);
      #pragma unroll
      for(int j=0;j<8;j++) {
        float activation=__bfloat162float(input[j]);
        gate[rr]=fmaf(fp4_value(g>>(4*j))*gsf,activation,gate[rr]);
        up[rr]=fmaf(fp4_value(u>>(4*j))*usf,activation,up[rr]);
      }
    }
  }
  float global=gs[expert];
  #pragma unroll
  for(int rr=0;rr<ROWS;rr++) {
    float g=warp_sum(gate[rr])*global,u=warp_sum(up[rr])*global;
    if(lane==0) {
      g=__bfloat162float(__float2bfloat16_rn(g));
      u=__bfloat162float(__float2bfloat16_rn(u));
      // Match the fused BF16-activation SwiGLU used by SGLang.
      out[(long)slot*I+row0+rr]=__float2bfloat16_rn(g/(1.0f+__expf(-g))*u);
    }
  }
}

template<int WARPS=8,int ROWS=4>
__global__ void fp4_down_kernel(const __nv_bfloat16* __restrict__ x,
    const uint32_t* __restrict__ packed,const unsigned char* __restrict__ sf,
    const float* __restrict__ gs,const int* __restrict__ ids,
    __nv_bfloat16* __restrict__ out,int start) {
  constexpr int K=4096,N=4096;
  int slot=blockIdx.y,expert=ids[slot]-start;
  if(expert<0 || expert>=32)return;
  int lane=threadIdx.x&31,warp=threadIdx.x>>5;
  int row0=(blockIdx.x*WARPS+warp)*ROWS;
  float acc[ROWS]={};
  #pragma unroll 1
  for(int kk=0;kk<K;kk+=256) {
    __nv_bfloat16 input[8];
    #pragma unroll
    for(int j=0;j<8;j++)input[j]=x[(long)slot*K+kk+lane*8+j];
    #pragma unroll
    for(int rr=0;rr<ROWS;rr++) {
      int row=row0+rr;
      uint32_t codes=packed[((long)expert*N+row)*(K/8)+kk/8+lane];
      float scale=fp8_scale(sf[((long)expert*N+row)*(K/16)+kk/16+lane/2]);
      #pragma unroll
      for(int j=0;j<8;j++)acc[rr]=fmaf(fp4_value(codes>>(4*j))*scale,__bfloat162float(input[j]),acc[rr]);
    }
  }
  float global=gs[expert];
  #pragma unroll
  for(int rr=0;rr<ROWS;rr++) {
    float value=warp_sum(acc[rr])*global;
    if(lane==0)out[(long)slot*N+row0+rr]=__float2bfloat16_rn(value);
  }
}

void achita_gate_up(torch::Tensor x,torch::Tensor w,torch::Tensor sf,torch::Tensor gs,
                    torch::Tensor ids,torch::Tensor out,int64_t start) {
  c10::cuda::CUDAGuard guard(x.device());
  auto stream=c10::cuda::getCurrentCUDAStream(x.get_device()).stream();
  fp4_gate_up_kernel<<<dim3(4096/32,8),256,0,stream>>>(
    (const __nv_bfloat16*)x.data_ptr(),(const uint32_t*)w.data_ptr(),
    (const unsigned char*)sf.data_ptr(),gs.data_ptr<float>(),ids.data_ptr<int>(),
    (__nv_bfloat16*)out.data_ptr(),int(start));
}
void achita_down(torch::Tensor x,torch::Tensor w,torch::Tensor sf,torch::Tensor gs,
                  torch::Tensor ids,torch::Tensor out,int64_t start) {
  c10::cuda::CUDAGuard guard(x.device());
  auto stream=c10::cuda::getCurrentCUDAStream(x.get_device()).stream();
  fp4_down_kernel<<<dim3(4096/32,8),256,0,stream>>>(
    (const __nv_bfloat16*)x.data_ptr(),(const uint32_t*)w.data_ptr(),
    (const unsigned char*)sf.data_ptr(),gs.data_ptr<float>(),ids.data_ptr<int>(),
    (__nv_bfloat16*)out.data_ptr(),int(start));
}
'''

_achita_fp4_base_prepare = prepare
_achita_fp4_base_kernel = kernel_fn


def prepare(weights):
    state = _achita_fp4_base_prepare(weights)
    name = 'achita_nvfp4_gemv_' + hashlib.sha256(_ACHITA_FP4_CUDA.encode()).hexdigest()[:12]
    state['achita_fp4_ext'] = load_inline(
        name, _ACHITA_FP4_CPP, cuda_sources=_ACHITA_FP4_CUDA,
        functions=['achita_gate_up', 'achita_down'], no_implicit_headers=True,
        with_pytorch_error_handling=False,
        extra_cuda_cflags=['-O3', '-std=c++20', '-lineinfo', '-gencode=arch=compute_90a,code=sm_90a'])
    # Clone only the local bank, retaining the exact original dtypes and bits.
    state['achita_fp4_raw'] = {key: weights[key][:32].clone() for key in
                             ('w13', 'w13_sf', 'w13_gs', 'w2', 'w2_sf', 'w2_gs')}
    return state


def kernel_fn(data, state):
    cfg = data['config']
    x = data['hidden_states']
    if (x.shape == (1, 4096) and cfg['d_expert'] == 4096
            and cfg['local_start'] == 0 and cfg['local_count'] == 32
            and x.is_contiguous() and data['topk_ids'].is_contiguous()):
        raw = state['achita_fp4_raw']
        ext = state['achita_fp4_ext']
        intermediate = torch.empty((8, 4096), device=x.device, dtype=torch.bfloat16)
        y = torch.empty((1, 8, 4096), device=x.device, dtype=torch.bfloat16)
        ids = data['topk_ids']
        ext.achita_gate_up(x, raw['w13'], raw['w13_sf'], raw['w13_gs'], ids, intermediate, 0)
        ext.achita_down(intermediate, raw['w2'], raw['w2_sf'], raw['w2_gs'], ids, y, 0)
        local_ids = state['banks'][(0, 32)]['expert_map'][ids]
        return fused_post_expert_norm_weighted_sum(
            y, data['topk_weights'], norm_weight=state['norm_w'], eps=POST_NORM_EPS,
            weight_first=True, slot_expert_ids=local_ids)
    return _achita_fp4_base_kernel(data, state)
