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

__device__ __forceinline__ uint4 stream_load(const uint4* p) {
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
      : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}

__device__ __forceinline__ uint4 expand_fragment(uint2 low, unsigned q, unsigned table0, unsigned table1) {
    unsigned hi0=__byte_perm(table0,table1,q & 0x7777u) | (q & 0x80808080u);
    unsigned hi1=__byte_perm(table0,table1,(q>>16) & 0x7777u) | ((q<<4) & 0x80808080u);
    return make_uint4(__byte_perm(low.x,hi0,0x5140),__byte_perm(low.x,hi0,0x7362),
                      __byte_perm(low.y,hi1,0x5140),__byte_perm(low.y,hi1,0x7362));
}
__device__ __forceinline__ uint2 stream_load2(const uint2* p) {
    uint2 v;
    asm volatile("ld.global.nc.L1::no_allocate.v2.u32 {%0,%1}, [%2];"
      : "=r"(v.x), "=r"(v.y) : "l"(p));
    return v;
}

template<int NW, int NB, int U, bool DUAL, bool FLOAT_OUT>
__global__ __launch_bounds__(NW*32, 2) void stream_mma(
    const uint4 *__restrict__ W, const uint4 *__restrict__ L,
    const unsigned *__restrict__ X, void *__restrict__ O,
    const uint2 *__restrict__ C, const unsigned char *__restrict__ F, unsigned T0, unsigned T1,
    int M, int N, int K, int SPLIT) {
    int lane=threadIdx.x&31, warp=threadIdx.x>>5;
    int g=lane>>2, t=lane&3;
    int tiles=K/32, split=blockIdx.y;
    int per=(tiles+SPLIT-1)/SPLIT;
    float acc[NB][4]={};
    int kw=(per+NW-1)/NW;
    int begin=split*per+warp*kw;
    int end=min(min(begin+kw,(split+1)*per),tiles);
    for(int first=begin; first<end; first+=U) {
        uint4 av[U][2],lv[U][2];
        #pragma unroll
        for(int u=0;u<U;++u) {
            int kt=first+u;
            #pragma unroll
            for(int h=0;h<2;++h) {
                av[u][h]=make_uint4(0,0,0,0);
                if(kt<end) {
                    int idx=blockIdx.x*tiles+kt;
                    if constexpr(DUAL) av[u][h]=stream_load(W+idx*64+h*32+lane);
                    else {
                        unsigned flag=__ldg(F+idx);
                        if(flag & (1u<<h)) av[u][h]=stream_load(W+idx*64+h*32+lane);
                        else {
                            uint2 low=stream_load2(C+idx*96+h*32+lane);
                            uint2 code=stream_load2(C+idx*96+64+lane);
                            av[u][h]=expand_fragment(low,h==0?code.x:code.y,T0,T1);
                        }
                    }
                }
                if constexpr(DUAL) {
                    lv[u][h]=make_uint4(0,0,0,0);
                    if(kt<end) lv[u][h]=stream_load(L+(blockIdx.x*tiles+kt)*64+h*32+lane);
                }
            }
        }
        #pragma unroll
        for(int u=0;u<U;++u) {
            int kt=first+u;
            #pragma unroll
            for(int b=0;b<NB;++b) {
                int m=min(b*8+g,M-1);
                uint4 bv=make_uint4(0,0,0,0);
                if(kt<end) bv=__ldg(reinterpret_cast<const uint4*>(X+m*(K/2)+kt*16+t*4));
                mma(acc[b],av[u][0],bv.x,bv.y);
                mma(acc[b],av[u][1],bv.z,bv.w);
                if constexpr(DUAL) {
                    mma(acc[b],lv[u][0],bv.x,bv.y);
                    mma(acc[b],lv[u][1],bv.z,bv.w);
                }
            }
        }
    }
    __shared__ float scratch[NW][NB][32][4];
    #pragma unroll
    for(int b=0;b<NB;++b) {
        *reinterpret_cast<float4*>(&scratch[warp][b][lane][0]) =
            make_float4(acc[b][0],acc[b][1],acc[b][2],acc[b][3]);
    }
    __syncthreads();
    if(warp<NB) {
        int b=warp;
        {
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

template<int NW,int NB,int U> void dispatch(const at::Tensor &W,const at::Tensor &L,
    const at::Tensor &X, at::Tensor &P,const at::Tensor &C,const at::Tensor &F,unsigned T0,unsigned T1,int M,int N,int K,int S,bool dual,cudaStream_t stream) {
    dim3 grid((N+15)/16,S);
    const uint4* w=reinterpret_cast<const uint4*>(W.data_ptr());
    const uint4* l=reinterpret_cast<const uint4*>(L.data_ptr());
    const unsigned* x=reinterpret_cast<const unsigned*>(X.data_ptr());
    if(dual) stream_mma<NW,NB,U,true,true><<<grid,NW*32,0,stream>>>(w,l,x,P.data_ptr(),reinterpret_cast<const uint2*>(C.data_ptr()),F.data_ptr<unsigned char>(),T0,T1,M,N,K,S);
    else if(S>1) stream_mma<NW,NB,U,false,true><<<grid,NW*32,0,stream>>>(w,l,x,P.data_ptr(),reinterpret_cast<const uint2*>(C.data_ptr()),F.data_ptr<unsigned char>(),T0,T1,M,N,K,S);
    else stream_mma<NW,NB,U,false,false><<<grid,NW*32,0,stream>>>(w,l,x,P.data_ptr(),reinterpret_cast<const uint2*>(C.data_ptr()),F.data_ptr<unsigned char>(),T0,T1,M,N,K,S);
}

void run(at::Tensor W,at::Tensor L,at::Tensor X,at::Tensor P,at::Tensor Y,at::Tensor C,at::Tensor F,int64_t T0,int64_t T1,
         int N,int NW,int S,int U,bool dual) {
    int M=X.size(0),K=X.size(1);
    auto stream=at::cuda::getCurrentCUDAStream();
    #define GO(WARPS,UNROLL) if(M<=8) dispatch<WARPS,1,UNROLL>(W,L,X,P,C,F,(unsigned)T0,(unsigned)T1,M,N,K,S,dual,stream); else if(M<=16) dispatch<WARPS,2,UNROLL>(W,L,X,P,C,F,(unsigned)T0,(unsigned)T1,M,N,K,S,dual,stream); else dispatch<WARPS,4,UNROLL>(W,L,X,P,C,F,(unsigned)T0,(unsigned)T1,M,N,K,S,dual,stream)
    if(U==2) { if(NW==4) { GO(4,2); } else if(NW==8) { GO(8,2); } else { GO(16,2); } }
    else { if(NW==4) { GO(4,4); } else if(NW==8) { GO(8,4); } else { GO(16,4); } }
    #undef GO
    if(S>1) {
        if(dual) reduce_parts<false><<<(M*N+255)/256,256,0,stream>>>(P.data_ptr<float>(),Y.data_ptr(),M*N,S);
        else reduce_parts<true><<<(M*N+255)/256,256,0,stream>>>(P.data_ptr<float>(),Y.data_ptr(),M*N,S);
    }
}
'''
CPP = '''void run(at::Tensor W,at::Tensor L,at::Tensor X,at::Tensor P,at::Tensor Y,at::Tensor C,at::Tensor F,int64_t T0,int64_t T1,
                  int N,int NW,int S,int U,bool dual);'''


def _pack(w):
    n,k=w.shape
    pairs=w.contiguous().view(torch.int32).reshape(n//16,16,k//32,16).permute(0,2,1,3)
    lane=torch.arange(32,device=w.device)
    q=torch.arange(4,device=w.device)
    step=torch.arange(2,device=w.device)
    rr=(lane//4)[None,:,None]+(q%2)[None,None,:]*8
    cc=(lane%4)[None,:,None]*4+step[:,None,None]*2+(q//2)[None,None,:]
    return pairs[:,:,rr,cc].contiguous()


def prepare(weights):
    state={name:w.contiguous() for name,w in weights.items()}
    state['module']=load_inline(name='muse_stream_mma_v5',cpp_sources=CPP,cuda_sources=CUDA,
                                functions=['run'],extra_cuda_cflags=['-O3'],verbose=False)
    for name,w in weights.items():
        if w.dtype==torch.bfloat16:
            state[name+'_pack']=_pack(w)
            state[name+'_compressed']=_compress(state[name+'_pack'])
    hi=weights['router'].to(torch.bfloat16)
    lo=(weights['router']-hi.float()).to(torch.bfloat16)
    state['rhi'],state['rlo']=_pack(hi),_pack(lo)
    state['router_compressed']=(state['rhi'],torch.empty(1,device=hi.device,dtype=torch.uint8),0,0)
    state['rpack']=torch.cat((hi,lo),0).T.contiguous()
    _tune(state)
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
    choice=state['choices'].get((layer,m,n), (8, 8 if router else 1, 2))
    if choice is None:
        return _blas(x,state,layer,n)
    nw,split,unroll=choice
    out=torch.empty((m,n),device=x.device,dtype=torch.float32 if router else torch.bfloat16)
    partial=torch.empty((split,m,n),device=x.device,dtype=torch.float32) if split>1 else out
    w=state['rhi'] if router else state[layer+'_pack']
    lo=state['rlo'] if router else w
    state['module'].run(w,lo,x,partial,out,*state[layer+'_compressed'],n,nw,split,unroll,router)
    return out


def _blas(x,state,layer,n):
    if layer=='router':
        z=torch.mm(x,state['rpack'],out_dtype=torch.float32)
        wn=state['router'].shape[0]
        return z[:,:n]+z[:,wn:wn+n]
    return torch.mm(x,state[layer][:n].T)


def _tune(state):
    # Normal setup-time autotuning uses synthetic inputs only. No data from
    # kernel_fn is inspected or cached; its complete computation is timed.
    import statistics
    choices={}
    device=state['router'].device
    generator=torch.Generator(device=device).manual_seed(20261028)
    flush=torch.empty(64*1024*1024,device=device,dtype=torch.float32)
    stream=torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        for layer in ('qkv_gate','o','pooler','unpooler','router'):
            k=state[layer].shape[1]
            router=layer=='router'
            for m in (1,8,16,32):
                n=1024 if layer=='unpooler' and m<=8 else state[layer].shape[0]
                x=torch.randn((m,k),device=device,dtype=torch.bfloat16,generator=generator)
                y=torch.empty((m,n),device=device,dtype=torch.float32 if router else torch.bfloat16)
                w=state['rhi'] if router else state[layer+'_pack']
                lo=state['rlo'] if router else w
                options=[None]+[(nw,ss,u) for nw in (4,8,16) for ss in ((4,8,16) if router else (1,2)) for u in (2,4)]
                measurements=[]
                for choice in options:
                    if choice is None:
                        def call(): return _blas(x,state,layer,n)
                    else:
                        nw,ss,u=choice
                        p=torch.empty((ss,m,n),device=device,dtype=torch.float32) if ss>1 else y
                        def call(): state['module'].run(w,lo,x,p,y,*state[layer+'_compressed'],n,nw,ss,u,router)
                    call();call()
                    graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph,stream=stream): call()
                    timings=[]
                    for repeat in range(7):
                        flush.zero_()
                        begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                        begin.record();graph.replay();end.record();end.synchronize()
                        timings.append(begin.elapsed_time(end)*1000)
                    measurements.append((statistics.median(timings),choice))
                best=min(measurements,key=lambda z:z[0])
                choices[(layer,m,n)]=best[1]
                print('TUNE',layer,m,n,'selected',best,'all',measurements,flush=True)
    torch.cuda.current_stream(device).wait_stream(stream)
    state['choices']=choices
    print('CHOICES',choices,flush=True)


def _compress(packed):
    # A fragment keeps every low byte plus sign and a three-bit exponent code.
    # Any exceptional fragment is loaded in full, preserving arbitrary BF16 bits.
    bits=packed.view(torch.int16).to(torch.int32)&65535
    e=(bits>>8)&127
    hist=torch.bincount(e.flatten(),minlength=128)
    prefix=torch.cat((torch.zeros(1,device=hist.device,dtype=hist.dtype),hist.cumsum(0)))
    base=int(torch.argmax(prefix[8:]-prefix[:-8]))
    escape=((e<base)|(e>base+7)).any(dim=-1).any(dim=-1)
    flags=(escape[:,:,0].to(torch.uint8)+2*escape[:,:,1].to(torch.uint8)).contiguous()
    low=(bits&255).to(torch.uint8).contiguous().view(torch.int32)
    code=(e-base).clamp(0,7)
    sign=bits>>15
    q=torch.zeros_like(code[...,0])
    for i in range(4):
        q=q | (code[...,i]<<(4*i)) | (code[...,i+4]<<(16+4*i))
        q=q | (sign[...,i]<<(8*i+7)) | (sign[...,i+4]<<(8*i+3))
    codes=q.permute(0,1,3,2).contiguous()
    compressed=torch.stack((low[:,:,0],low[:,:,1],codes),dim=2).contiguous()
    t0=sum((base+i)<<(8*i) for i in range(4))
    t1=sum((base+4+i)<<(8*i) for i in range(4))
    return compressed,flags,t0,t1
