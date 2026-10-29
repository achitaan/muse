"""Add exact native resource admission to the frozen two-kernel Attention draft."""
import ast
import hashlib
import json
from pathlib import Path

from build_attention_external_finalize import ROOT,OUT,freeze

DRAFT='4cbe02668aaf52bd5c514f470c1cec3b7b433932fbbed5a87abc7078daa74432'
RESOURCE=r'''
std::vector<int64_t> external_resources() {
  auto old_main = attn_kernel_fp8pv<8,true,0>;
  auto new_main = attn_kernel_fp8pv<8,true,16>;
  auto final = attn_external_finalize8;
  const int sm = static_cast<int>(sizeof(Smem<8,true>))+1024;
  cudaFuncAttributes a{},b{},c{};
  int ao=0,bo=0,co=0,device=0,sms=0;
  TORCH_CHECK((cudaFuncSetAttribute(old_main,cudaFuncAttributeMaxDynamicSharedMemorySize,sm)==cudaSuccess),"old main shared-size query failed");
  TORCH_CHECK((cudaFuncSetAttribute(new_main,cudaFuncAttributeMaxDynamicSharedMemorySize,sm)==cudaSuccess),"new main shared-size query failed");
  TORCH_CHECK((cudaFuncGetAttributes(&a,old_main)==cudaSuccess),"old main attributes failed");
  TORCH_CHECK((cudaFuncGetAttributes(&b,new_main)==cudaSuccess),"new main attributes failed");
  TORCH_CHECK((cudaFuncGetAttributes(&c,final)==cudaSuccess),"parallel finalizer attributes failed");
  TORCH_CHECK((cudaOccupancyMaxActiveBlocksPerMultiprocessor(&ao,old_main,Cfg<8,true>::NTX,sm)==cudaSuccess),"old main occupancy failed");
  TORCH_CHECK((cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bo,new_main,Cfg<8,true>::NTX,sm)==cudaSuccess),"new main occupancy failed");
  TORCH_CHECK((cudaOccupancyMaxActiveBlocksPerMultiprocessor(&co,final,128,0)==cudaSuccess),"finalizer occupancy failed");
  TORCH_CHECK((cudaGetDevice(&device)==cudaSuccess),"device query failed");
  TORCH_CHECK((cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,device)==cudaSuccess),"SM query failed");
  return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,ao,
          b.numRegs,(int64_t)b.localSizeBytes,(int64_t)b.sharedSizeBytes,bo,
          sm,c.numRegs,(int64_t)c.localSizeBytes,(int64_t)c.sharedSizeBytes,co,
          sms,64,Cfg<8,true>::NTX,128,16};
}
'''


def main():
    raw=(OUT/'submission.py').read_bytes();assert hashlib.sha256(raw).hexdigest()==DRAFT
    tree=ast.parse(raw);lines=raw.splitlines(keepends=True);edits=[]
    def replace(node,value):
        a=sum(map(len,lines[:node.lineno-1]))+node.col_offset
        b=sum(map(len,lines[:node.end_lineno-1]))+node.end_col_offset
        edits.append((a,b,value))
    values={n.targets[0].id:n for n in tree.body if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name)}
    cuda=ast.literal_eval(values['CUDA_SRC'].value)
    cpp=ast.literal_eval(values['CPP_SRC'].value)
    new_cuda=cuda+RESOURCE
    new_cpp=cpp+'\n#include <vector>\nstd::vector<int64_t> external_resources();\n'
    replace(values['CUDA_SRC'].value,repr(new_cuda).encode())
    replace(values['CPP_SRC'].value,repr(new_cpp).encode())
    build=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_build')
    old_build=b'\n'.join(raw.splitlines()[build.lineno-1:build.end_lineno]).decode()
    assert old_build.count('return load_inline(')==old_build.count('functions=["run"]')==1
    new_build=old_build.replace('return load_inline(','module = load_inline(',1).replace('functions=["run"]','functions=["run", "external_resources"]',1)
    new_build+='''
    resources = list(module.external_resources())
    assert len(resources) == 18
    assert 0 < resources[0] <= 80 and resources[1:3] == [0, 0] and resources[3] >= 2
    assert 0 < resources[4] <= 80 and resources[5:7] == [0, 0] and resources[7] >= 2
    assert resources[8] == 114688 and 0 < resources[9] <= 64 and resources[10:12] == [0, 0]
    assert resources[12] >= 1 and resources[13] >= 132 and resources[14:] == [64, 384, 128, 16]
    print("ATTN_EXTERNAL_RESOURCES", *resources, flush=True)
    return module'''
    replace(build,new_build.encode())
    candidate=raw
    for a,b,value in sorted(edits,reverse=True):candidate=candidate[:a]+value+candidate[b:]
    revised=ast.parse(candidate)
    for a,b in zip(tree.body,revised.body):
        if ast.dump(a)!=ast.dump(b):
            assert a in (values['CUDA_SRC'],values['CPP_SRC'],build)
    restored=revised
    for i,node in enumerate(restored.body):
        if isinstance(node,ast.FunctionDef) and node.name=='_build':restored.body[i]=build
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
            if node.targets[0].id in ('CUDA_SRC','CPP_SRC'):restored.body[i]=values[node.targets[0].id]
    assert ast.dump(restored)==ast.dump(tree)
    source_sha=hashlib.sha256(candidate).hexdigest();cuda_sha=hashlib.sha256(new_cuda.encode()).hexdigest()
    folder=OUT.parent/'admission-v2'
    freeze(folder/'submission.py',candidate);freeze(folder/'kernel.cu',new_cuda.encode())
    freeze(folder/'binding.json',(json.dumps(dict(source_sha256=source_sha,CUDA_sha256=cuda_sha,draft_sha256=DRAFT,
        bytes=len(candidate),only_native_resource_query_and_build_admission_added=True,
        exact_draft_AST_restored=True,numerical_device_prefix_unchanged=True,
        required_old_and_new_main_dynamic_bytes=114688,required_old_and_new_main_blocks_per_SM=2,
        required_old_and_new_main_register_limit=80,required_local_and_static_bytes=0,
        finalizer_threads=128,mode=16,CPU_only=True,GPU_jobs=0,
        performance_gate=dict(affected_case_GM=1.02,normalized_affected_GM=1.02,
                              minimum_each_affected_normalized_case=1.01,minimum_weighted_track_ratio=1.005,
                              controls_range=[.95,1.05]),
        not_admitted_for_GPU_until='fresh evaluator identity, independent result guard with fault injection, source/precision inverse proof and cached baseline binding'),indent=2)+'\n').encode())
    print(json.dumps(dict(source_sha256=source_sha,CUDA_sha256=cuda_sha,bytes=len(candidate),GPU_jobs=0)))


if __name__=='__main__':main()
