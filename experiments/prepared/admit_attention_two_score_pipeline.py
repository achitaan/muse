"""Freeze native admission and bounded dispatch for the compiled L8 pipeline."""
import ast
import hashlib
import json

from build_attention_two_score_pipeline import ROOT, OUT, PARENT, freeze, once

RESOURCE = r'''
std::vector<int64_t> two_score_resources() {
  auto old_main = attn_kernel_fp8pv<8,true,0>;
  auto next_main = attn_kernel_two_score<8,true,0>;
  const int sm = static_cast<int>(sizeof(Smem<8,true>))+1024;
  cudaFuncAttributes a{},b{};
  int ao=0,bo=0,device=0,sms=0;
  TORCH_CHECK((cudaFuncSetAttribute(old_main,cudaFuncAttributeMaxDynamicSharedMemorySize,sm)==cudaSuccess),"old main shared-size admission failed");
  TORCH_CHECK((cudaFuncSetAttribute(next_main,cudaFuncAttributeMaxDynamicSharedMemorySize,sm)==cudaSuccess),"pipeline shared-size admission failed");
  TORCH_CHECK((cudaFuncGetAttributes(&a,old_main)==cudaSuccess),"old main attributes failed");
  TORCH_CHECK((cudaFuncGetAttributes(&b,next_main)==cudaSuccess),"pipeline attributes failed");
  TORCH_CHECK((cudaOccupancyMaxActiveBlocksPerMultiprocessor(&ao,old_main,Cfg<8,true>::NTX,sm)==cudaSuccess),"old main occupancy failed");
  TORCH_CHECK((cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bo,next_main,Cfg<8,true>::NTX,sm)==cudaSuccess),"pipeline occupancy failed");
  TORCH_CHECK((cudaGetDevice(&device)==cudaSuccess),"device query failed");
  TORCH_CHECK((cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,device)==cudaSuccess),"SM query failed");
  return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,ao,
          b.numRegs,(int64_t)b.localSizeBytes,(int64_t)b.sharedSizeBytes,bo,
          sm,sms,Cfg<8,true>::NTX,Cfg<8,true>::STAGES,32,232};
}
'''


def main():
    raw=(ROOT/'best/attention_checked.py').read_bytes()
    assert hashlib.sha256(raw).hexdigest()==PARENT
    tree=ast.parse(raw);lines=raw.splitlines(keepends=True)
    values={n.targets[0].id:n for n in tree.body if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name)}
    prototype=(OUT.parent/'v2-register-repair/kernel.cu').read_text()
    assert hashlib.sha256(prototype.encode()).hexdigest()=='ed8f90ee809f72f4acfdf3b0d6bada9cd1d1e0c7f0be284f5da892a0b6a8d957'
    cuda=once(prototype,'  static int occ_c = 0, occ_f = 0, nsm = 0;',
        '  static int occ_c = 0, occ_f = 0, nsm = 0;\n  static int pipeline_occ = 0, pipeline_regs = 0;')
    init='    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ_f, attn_kernel_fp8pv<L, true>, Cfg<L, true>::NTX, sm_f);'
    extra=r'''
    if constexpr (L == 8) {
      auto pipe = attn_kernel_two_score<8,true,0>;
      cudaFuncAttributes attributes{};
      TORCH_CHECK((cudaFuncSetAttribute(pipe,cudaFuncAttributeMaxDynamicSharedMemorySize,sm_f)==cudaSuccess),"pipeline shared-memory setup failed");
      TORCH_CHECK((cudaFuncGetAttributes(&attributes,pipe)==cudaSuccess),"pipeline function query failed");
      TORCH_CHECK((cudaOccupancyMaxActiveBlocksPerMultiprocessor(&pipeline_occ,pipe,Cfg<8,true>::NTX,sm_f)==cudaSuccess),"pipeline occupancy query failed");
      pipeline_regs = attributes.localSizeBytes == 0 && attributes.sharedSizeBytes == 0 ? attributes.numRegs : 0;
    }
'''
    cuda=once(cuda,init,init+extra)
    launch=r'''  if constexpr (L == 8) {
    const int pairs = p.Hkv / 2;
    // This grid performs a global join. Every CTA must fit simultaneously;
    // the register pool proof additionally requires exactly168 initial GPRs.
    if (p.flat && q.coop && p.window == 0 && p.B > 1 && p.B <= 64 &&
        p.nc == 1 && !pdl && !p.dbg && !cl && pipeline_occ == 1 &&
        pipeline_regs == 168 && pairs > 0 && pairs <= nsm) {
      const int splits = nsm / pairs;
      q.n_splits = splits;
      q.ns_magic = splits >= 2 && splits <= 1024 ?
          (unsigned)(((1ull << 32) + (uint64_t)splits - 1) / (uint64_t)splits) : 0u;
      const dim3 pipeline_grid(splits,1,pairs);
      TORCH_CHECK((long long)pipeline_grid.x*pipeline_grid.z <= (long long)pipeline_occ*nsm,"nonresident pipeline grid");
      attn_kernel_two_score<8,true,0><<<pipeline_grid,Cfg<8,true>::NTX,sm_f,stream>>>(q);
      TORCH_CHECK((cudaGetLastError()==cudaSuccess),"attention pipeline launch failed");
      return;
    }
  }
'''
    cuda=once(cuda,'  if (cl || pdl) {\n    // cluster dims and / or programmatic stream serialization',
        launch+'  if (cl || pdl) {\n    // cluster dims and / or programmatic stream serialization')
    cuda+=RESOURCE
    cpp=ast.literal_eval(values['CPP_SRC'].value)+'\n#include <vector>\nstd::vector<int64_t> two_score_resources();\n'
    build=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_build')
    old_build='\n'.join(raw.decode().splitlines()[build.lineno-1:build.end_lineno])
    new_build=once(once(old_build,'return load_inline(','module = load_inline('),
        'functions=["run"]','functions=["run", "two_score_resources"]')
    new_build+='''
    resources = list(module.two_score_resources())
    assert len(resources) == 14
    assert 0 < resources[0] <= 80 and resources[1:3] == [0, 0] and resources[3] >= 2
    assert resources[4:9] == [168, 0, 0, 1, 114688]
    assert resources[9] >= 132 and resources[10:] == [384, 4, 32, 232]
    print("ATTN_TWO_SCORE_RESOURCES", *resources, flush=True)
    return module'''
    edits=[]
    for node,value in ((values['CUDA_SRC'].value,repr(cuda)),(values['CPP_SRC'].value,repr(cpp)),(build,new_build)):
        a=sum(map(len,lines[:node.lineno-1]))+node.col_offset
        b=sum(map(len,lines[:node.end_lineno-1]))+node.end_col_offset
        edits.append((a,b,value.encode()))
    candidate=raw
    for a,b,value in sorted(edits,reverse=True):candidate=candidate[:a]+value+candidate[b:]
    restored=ast.parse(candidate)
    for i,node in enumerate(restored.body):
        if isinstance(node,ast.FunctionDef) and node.name=='_build':restored.body[i]=build
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in ('CPP_SRC','CUDA_SRC'):
            restored.body[i]=values[node.targets[0].id]
    assert ast.dump(restored)==ast.dump(tree)
    assert cuda.split('template <int L>\nvoid launch(')[0]==prototype.split('template <int L>\nvoid launch(')[0]
    out=OUT.parent/'admission-v3'
    report=dict(source_sha256=hashlib.sha256(candidate).hexdigest(),CUDA_sha256=hashlib.sha256(cuda.encode()).hexdigest(),
        parent_source_sha256=PARENT,bytes=len(candidate),CPU_only=True,GPU_jobs=0,
        compiled_device_prefix_exact=True,other_Python_AST_exact=True,
        performance_gate=dict(affected_case_GM=1.02,normalized_affected_GM=1.02,
            minimum_each_affected_normalized_case=1.01,minimum_weighted_track_ratio=1.005,controls_range=[.95,1.05]),
        scope='L8 full,B2..64,flat,nc1,retained coop enabled,no PDL/debug/clusters;168 native registers,zero local/static,one CTA/SM',
        not_admitted_for_GPU_until='Independent source/model/coverage proof, fault-injected guard, current evaluator identity, cached baseline binding')
    freeze(out/'submission.py',candidate);freeze(out/'kernel.cu',cuda.encode())
    freeze(out/'binding.json',(json.dumps(report,indent=2)+'\n').encode())
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
