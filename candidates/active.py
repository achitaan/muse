# okc: problem=muse_bf16_proj
"""Losslessly repacked BF16 Tensor Core streaming for small token batches.

The fragment layout follows NVIDIA PTX mma.m16n8k16 documentation. All products
accumulate in FP32; router hi/lo is the decomposition allowed by the task.
"""
import torch
from torch.utils.cpp_extension import load_inline

CUDA = r'''
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

__device__ __forceinline__ void mma(float *d, uint4 a, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b0), "r"(b1));
}

template<int NW, int NB, bool DUAL, bool FLOAT_OUT>
__global__ __launch_bounds__(NW*32) void stream_mma(
    const uint4 *__restrict__ W, const uint4 *__restrict__ L,
    const unsigned *__restrict__ X, void *__restrict__ O,
    int M, int N, int K, int SPLIT) {
    int lane=threadIdx.x&31, warp=threadIdx.x>>5;
    int g=lane>>2, t=lane&3;
    int tiles=K/16, split=blockIdx.y;
    int per=(tiles+SPLIT-1)/SPLIT;
    float acc[NB][4]={};
    for(int kt=split*per+warp; kt<(split+1)*per && kt<tiles; kt+=NW) {
        uint4 a=W[(blockIdx.x*tiles+kt)*32+lane];
        uint4 lo;
        if constexpr(DUAL) lo=L[(blockIdx.x*tiles+kt)*32+lane];
        #pragma unroll
        for(int b=0;b<NB;++b) {
            int m=b*8+g;
            unsigned b0=0,b1=0;
            if(m<M) { b0=X[m*(K/2)+kt*8+t]; b1=X[m*(K/2)+kt*8+t+4]; }
            mma(acc[b],a,b0,b1);
            if constexpr(DUAL) mma(acc[b],lo,b0,b1);
        }
    }
    __shared__ float scratch[NW][NB][32][4];
    #pragma unroll
    for(int b=0;b<NB;++b) {
        *reinterpret_cast<float4*>(&scratch[warp][b][lane][0]) =
            make_float4(acc[b][0],acc[b][1],acc[b][2],acc[b][3]);
    }
    __syncthreads();
    if(warp==0) {
        #pragma unroll
        for(int b=0;b<NB;++b) {
            float sum[4]={};
            #pragma unroll
            for(int w=0;w<NW;++w) {
                float4 v=*reinterpret_cast<float4*>(&scratch[w][b][lane][0]);
                sum[0]+=v.x; sum[1]+=v.y; sum[2]+=v.z; sum[3]+=v.w;
            }
            #pragma unroll
            for(int j=0;j<4;++j) {
                int n=blockIdx.x*16+g+(j>=2?8:0);
                int m=b*8+2*t+(j&1);
                if(n<N && m<M) {
                    int pos=(split*M+m)*N+n;
                    if constexpr(FLOAT_OUT) reinterpret_cast<float*>(O)[pos]=sum[j];
                    else reinterpret_cast<__nv_bfloat16*>(O)[pos]=__float2bfloat16_rn(sum[j]);
                }
            }
        }
    }
}

template<bool BF16> __global__ void reduce_parts(const float *P, void *O, int size, int S) {
    int i=blockIdx.x*256+threadIdx.x;
    if(i<size) {
        float v=0.f;
        for(int s=0;s<S;++s) v+=P[s*size+i];
        if constexpr(BF16) reinterpret_cast<__nv_bfloat16*>(O)[i]=__float2bfloat16_rn(v);
        else reinterpret_cast<float*>(O)[i]=v;
    }
}

template<int NW,int NB> void dispatch(const at::Tensor &W,const at::Tensor &L,
    const at::Tensor &X, at::Tensor &P,int M,int N,int K,int S,bool dual,cudaStream_t stream) {
    dim3 grid((N+15)/16,S);
    const uint4* w=reinterpret_cast<const uint4*>(W.data_ptr());
    const uint4* l=reinterpret_cast<const uint4*>(L.data_ptr());
    const unsigned* x=reinterpret_cast<const unsigned*>(X.data_ptr());
    if(dual) stream_mma<NW,NB,true,true><<<grid,NW*32,0,stream>>>(w,l,x,P.data_ptr(),M,N,K,S);
    else if(S>1) stream_mma<NW,NB,false,true><<<grid,NW*32,0,stream>>>(w,l,x,P.data_ptr(),M,N,K,S);
    else stream_mma<NW,NB,false,false><<<grid,NW*32,0,stream>>>(w,l,x,P.data_ptr(),M,N,K,S);
}

void run(at::Tensor W,at::Tensor L,at::Tensor X,at::Tensor P,at::Tensor Y,
         int N,int NW,int S,bool dual) {
    int M=X.size(0),K=X.size(1);
    auto stream=at::cuda::getCurrentCUDAStream();
    #define GO(WARPS) if(M<=8) dispatch<WARPS,1>(W,L,X,P,M,N,K,S,dual,stream); else if(M<=16) dispatch<WARPS,2>(W,L,X,P,M,N,K,S,dual,stream); else dispatch<WARPS,4>(W,L,X,P,M,N,K,S,dual,stream)
    if(NW==4) { GO(4); } else if(NW==8) { GO(8); } else { GO(16); }
    #undef GO
    if(S>1) {
        if(dual) reduce_parts<false><<<(M*N+255)/256,256,0,stream>>>(P.data_ptr<float>(),Y.data_ptr(),M*N,S);
        else reduce_parts<true><<<(M*N+255)/256,256,0,stream>>>(P.data_ptr<float>(),Y.data_ptr(),M*N,S);
    }
}
'''
CPP = '''void run(at::Tensor W,at::Tensor L,at::Tensor X,at::Tensor P,at::Tensor Y,
                  int N,int NW,int S,bool dual);'''


def _pack(w):
    n,k=w.shape
    # Pair adjacent BF16 values without changing their bits. A fragments are
    # contiguous uint4 per lane: (row g, row g+8) at K offsets 2t and 2t+8.
    pairs=w.contiguous().view(torch.int32).reshape(n//16,16,k//16,8).permute(0,2,1,3)
    lane=torch.arange(32,device=w.device)
    rr=(lane//4)[:,None]+torch.tensor([0,8,0,8],device=w.device)[None,:]
    cc=(lane%4)[:,None]+torch.tensor([0,0,4,4],device=w.device)[None,:]
    return pairs[:,:,rr,cc].contiguous()


def prepare(weights):
    state={name:w.contiguous() for name,w in weights.items()}
    state['module']=load_inline(name='muse_stream_mma_v1',cpp_sources=CPP,cuda_sources=CUDA,
                                functions=['run'],extra_cuda_cflags=['-O3'],verbose=False)
    for name,w in weights.items():
        if w.dtype==torch.bfloat16:state[name+'_pack']=_pack(w)
    hi=weights['router'].to(torch.bfloat16)
    lo=(weights['router']-hi.float()).to(torch.bfloat16)
    state['rhi'],state['rlo']=_pack(hi),_pack(lo)
    state['rpack']=torch.cat((hi,lo),0).T.contiguous()
    return state


def kernel_fn(data,state):
    x,cfg=data['x'],data['config']
    m,k=x.shape
    n,layer=cfg['n'],cfg['layer']
    router=layer=='router'
    if m>32:
        if router:
            z=torch.mm(x,state['rpack'],out_dtype=torch.float32)
            wn=state['router'].shape[0]
            return z[:,:n]+z[:,wn:wn+n]
        return torch.mm(x,state[layer][:n].T)
    nw=8
    split=8 if router else 1
    out=torch.empty((m,n),device=x.device,dtype=torch.float32 if router else torch.bfloat16)
    partial=torch.empty((split,m,n),device=x.device,dtype=torch.float32) if split>1 else out
    w=state['rhi'] if router else state[layer+'_pack']
    lo=state['rlo'] if router else w
    state['module'].run(w,lo,x,partial,out,n,nw,split,router)
    return out
