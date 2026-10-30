"""Package the four frozen projection twins and fixed norm/FPR handoffs."""

import ast
import base64
import hashlib
import json
import zlib

from pdl16_sources import HERE, ROOT, BASE_SHA, literal
from build_norm_pair_diag import extract, DENSE_SHA
from harness import append_harness

OUT=ROOT/'results/decode-moe-plan-20261030/handoff16-diag'

SPIN=r'''
// Prepare-only five-microsecond producer. All source values are copied by a
// preceding graph node; first-column sign bits are published only at its end.
__global__ void d16_spin_kernel(const unsigned short* src,unsigned short* dst,int m,int k,unsigned long long ns) {
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  if(threadIdx.x==0) {
    unsigned long long begin,now;
    asm volatile("mov.u64 %0, %%globaltimer;":"=l"(begin));
    do { asm volatile("mov.u64 %0, %%globaltimer;":"=l"(now)); } while(now-begin<ns);
  }
  __syncthreads();
  if(threadIdx.x<m) {
    const size_t i=(size_t)threadIdx.x*k;
    dst[i]=src[i]^0x8000u;
  }
}
int d16_spin_start(int64_t src,int64_t dst,int m,int k,int64_t stream,int64_t ns) {
  if(m<1||m>32||k<1||ns!=5000)return -70;
  cudaLaunchConfig_t c{};c.gridDim=dim3(1);c.blockDim=dim3(256);c.stream=(cudaStream_t)stream;
  return (int)cudaLaunchKernelEx(&c,d16_spin_kernel,(const unsigned short*)src,(unsigned short*)dst,m,k,(unsigned long long)ns);
}
'''

QUERY=r'''
template<class Kernel> static int64_t d16_fpr_attr(Kernel kernel,int64_t what) {
  cudaFuncAttributes a{};cudaError_t s=cudaFuncGetAttributes(&a,kernel);
  if(s!=cudaSuccess)return -1000-(int)s;
  if(what==0)return a.numRegs;
  if(what==1)return a.localSizeBytes;
  if(what==2)return a.sharedSizeBytes;
  if(what==3)return a.maxThreadsPerBlock;
  return -1;
}
int64_t d16_fpr_query(int64_t m,int64_t what) {
  if(m==16) {
    return d16_fpr_attr(&sk2_kernel<1,16,4,1,8,1,5,true,false,2,1,__nv_bfloat16,false,true>,what);
  }
  if(m==32) {
    return d16_fpr_attr(&sk2_kernel<2,32,4,1,8,1,3,true,false,2,1,__nv_bfloat16,false,true>,what);
  }
  return -2;
}
'''


def constants(src,names):
    found={n.targets[0].id:ast.get_source_segment(src,n)for n in ast.parse(src).body
        if isinstance(n,ast.Assign)and isinstance(n.targets[0],ast.Name)and n.targets[0].id in names}
    assert set(found)==set(names)
    return '\n'.join(found[name]for name in names)+'\n'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    dense=(ROOT/'best/verified.py').read_bytes();decode=(HERE/'decoder_pdli.py').read_bytes()
    chain=(HERE/'decoder_pdl_chain_draft.py').read_bytes();norm=(HERE/'decoder_pdln_draft.py').read_bytes()
    assert hashlib.sha256(dense).hexdigest()==DENSE_SHA and hashlib.sha256(decode).hexdigest()==BASE_SHA
    assert hashlib.sha256(chain).hexdigest()=='985abda3f1a91e6daca07a38e9f91a397b4942ef892285704ddbd1792bff9a40'
    assert hashlib.sha256(norm).hexdigest()=='7f5a03dd52e9c888fe7d9854c4acbe96b0b79bca90806283d0c49a58f77ca671'
    cs=chain.decode();ds=decode.decode();ns=norm.decode()
    values={n.targets[0].id:n.value for n in ast.parse(cs).body if isinstance(n,ast.Assign)and isinstance(n.targets[0],ast.Name)}
    cuda=literal(values['_P16_CUDA']);cpp=literal(values['_P16_CPP']);functions=literal(values['_P16_FUNCTIONS'])
    assert hashlib.sha256(cuda.encode()).hexdigest()=='09b371d0ffce2e93655e5ebdc188a5276e62c2217d62b63a3b8f139be49be220'
    bundle=cuda+SPIN;bundle_cpp=cpp+'\n#include <stdint.h>\nint d16_spin_start(int64_t,int64_t,int,int,int64_t,int64_t);\n'
    (OUT/'projection-original-prefix.cu').write_bytes(cuda.encode())
    (OUT/'projection-with-spin.cu').write_bytes(bundle.encode());(OUT/'projection.cpp').write_bytes(bundle_cpp.encode())
    dv={n.targets[0].id:n.value for n in ast.parse(ds).body if isinstance(n,ast.Assign)and isinstance(n.targets[0],ast.Name)}
    fpr=literal(dv['_fpr_CUDA']);fpr_cpp=literal(dv['_fpr_CPP'])
    assert fpr.encode()==(OUT.parent/'pdl16/fpr-parent.cu').read_bytes()
    fpr_query=fpr+QUERY
    (OUT/'fpr-original-prefix.cu').write_bytes(fpr.encode());(OUT/'fpr-with-query.cu').write_bytes(fpr_query.encode())
    app=append_harness('')+'\nimport hashlib\n'
    app+='def _d16_zsrc(s):\n import base64,zlib\n return zlib.decompress(base64.b64decode(s)).decode()\n'
    for key,value in {'_D16_CUDA':bundle,'_fpr_CUDA':fpr_query}.items():
        app+=key+' = _d16_zsrc('+repr(base64.b64encode(zlib.compress(value.encode(),9)).decode())+')\n'
    for key,value in {'_D16_CPP':bundle_cpp,'_D16_FUNCTIONS':list(functions)+['d16_spin_start'],
        '_D16_CUDA_SHA':hashlib.sha256(bundle.encode()).hexdigest(),
        '_fpr_CPP':fpr_cpp+'\n#include <stdint.h>\nint64_t d16_fpr_query(int64_t,int64_t);\n',
        '_D16_FPR_ORIGINAL_SHA':hashlib.sha256(fpr.encode()).hexdigest(),
        '_D16_PUBLIC_SHA':json.loads((OUT.parent/'pdl-chain/stock-norm-pinned/origin.json').read_text())['files']['norm_residual.py']['sha256']}.items():
        app+=key+' = '+repr(value)+'\n'
    app+=constants(cs,['_P16_M16_EXPORT','_P16_M16_INST','_P16_U16_INST'])
    app+=constants(ns,['_PDLN_EXPECTED_HEADERS','_PDLN_QUERY','_PDLN_NOTICE','_PDLN_SOURCE_SHA'])
    runtime_hash=hashlib.sha256((OUT.parent/'pdl-norm/upstream/runtime.cuh').read_bytes()).hexdigest()
    app+="_PDLN_EXPECTED_HEADERS['runtime.cuh'] = "+repr(runtime_hash)+'\n'
    app+=extract(ns,['pdln_derive','_pdln_build','_pdln_boundary'])
    app+=extract(cs,['_p16_project','_p16_unpool','_p16_pool_route','_p16_plans'])
    app+=constants(ds,['_fpr_GEOMETRIES'])
    app+='from torch.utils.cpp_extension import load_inline as _fpr_load_inline\n_fpr_ext = None\n_fpr_PDL = 0\n'
    app+=extract(ds,['_fpr_bfp_sk2_tileunits','_fpr_bfp_sk2_compress','_fpr_bfp_sk2_stream',
        '_fpr_bfp_sk2_table','_fpr_bfp__sk2_tokr','_fpr_bfp_fused_pooler_router','_fpr_prepare'])
    getter=extract(ds,['_fpr_get_ext'])
    old_name="'okc_fpr_f68f77c0d29cc56b_'"
    assert getter.count(old_name)==1
    getter=getter.replace(old_name,repr('okc_d16_fpr_'+hashlib.sha256(fpr_query.encode()).hexdigest()[:12]+'_'))
    old="functions=['sk2f', 'sk2_set_pdl']";assert getter.count(old)==1
    getter=getter.replace(old,"functions=['sk2f', 'sk2_set_pdl', 'd16_fpr_query']")
    app+=getter+extract(ds,['_pdli_equal','_pdli_probe'])
    app+=(HERE/'handoff16_diag_runtime.py').read_text(encoding='utf-8')
    raw=dense+app.encode();ast.parse(raw)
    assert raw.startswith(dense)and len(raw)<1_000_000
    (HERE/'dense_handoff16_diag.py').write_bytes(raw)
    record=dict(source_sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),Dense_parent_sha256=DENSE_SHA,
        Decode_parent_sha256=BASE_SHA,projection_original_CUDA_sha256=hashlib.sha256(cuda.encode()).hexdigest(),
        projection_diagnostic_CUDA_sha256=hashlib.sha256(bundle.encode()).hexdigest(),
        FPR_original_CUDA_sha256=hashlib.sha256(fpr.encode()).hexdigest(),
        FPR_with_query_CUDA_sha256=hashlib.sha256(fpr_query.encode()).hexdigest(),
        norm_CUDA_sha256=literal(next(n.value for n in ast.parse(ns).body if isinstance(n,ast.Assign)
            and getattr(n.targets[0],'id','')=='_PDLN_SOURCE_SHA')),
        changed_FPR_host='Only unique module name and additional diagnostic resource export; kernel source is an exact prefix.',
        fixed_families=['O16','O32','U16','U32','norm_FPR16_depth5','norm_FPR32_depth3'],
        GPU_jobs=0,enables_nothing=True,timed_Dense_parent_unchanged=True,
        synthetic_ceiling='O/U: one CTA,5000ns,late first-column writes after graph input copy; optimistic,not real layer overlap',
        real_chain='norm/FPR: parent PDL0,original norm+fixed PDL,early norm+same fixed PDL',
        FPR_query='Same fused kernel in all arms; PDL is its runtime integer, last template bool is FUSE=true.',
        gates='Finite bytes on changing normal/zero/outlier probes; >3% and >=1.5us both orders. Norm/FPR must also beat its late-signal PDL control by the same margins. No whole-layer claim.')
    (OUT/'source-binding.json').write_bytes((json.dumps(record,indent=2)+'\n').encode());print(json.dumps(record,indent=2))


if __name__=='__main__':main()
