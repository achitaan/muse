from pathlib import Path
import ast,hashlib,json,sys
sys.path.insert(0,str(Path('experiments/prepared').resolve()))
from local_nvrtc_compile import NVRTC,TAGS,STDINT,TOOL,PINNED
for name,h in PINNED.items(): assert hashlib.sha256((TOOL/'wheels'/name).read_bytes()).hexdigest()==h
raw=Path('best/attention_checked.py').read_bytes()
assert hashlib.sha256(raw).hexdigest()=='9e922c3555484081ca45a99a90284e26eccab181fb677700d8227a4b3402412f'
tree=ast.parse(raw)
src=next(n.value.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CUDA_SRC' for t in n.targets))
anchor='template <int L>\nvoid launch('
assert src.count(anchor)==1
prefix=src.split(anchor)[0]
removed=['#include <torch/extension.h>\n','#include <ATen/cuda/CUDAContext.h>\n','#include <cuda.h>\n']
for s in removed:
 assert prefix.count(s)==1
 prefix=prefix.replace(s,'')
flags=['-DINFINITY=__int_as_float(0x7f800000)','--gpu-architecture=sm_90a','--std=c++20','--generate-line-info','--device-as-default-execution-space','--ptxas-options=-v','--include-path='+str(TOOL/'nvidia/cu13/include'),'--include-path='+str(TOOL/'nvidia/cu13/include/cccl')]
old_drain='           // done before the next tile (no wgmma in flight across the loop edge)\n          wgmma_wait<0>();\n#pragma unroll\n          for (int e = 0; e < 32; ++e) reg_fence(od[e]);\n#pragma unroll\n          for (int e = 0; e < 4; ++e) reg_fence(lacc[e]);\n#pragma unroll\n          for (int e = 0; e < 8; ++e) asm volatile("" : "+r"(pa[e >> 2][e & 3]));'
assert prefix.count(old_drain)==1
original_prefix=prefix
prefix=prefix.replace(old_drain,'// CPU-only diagnostic: omit end-of-tile drain; not GPU-validated')
assert prefix.replace('// CPU-only diagnostic: omit end-of-tile drain; not GPU-validated',old_drain)==original_prefix
acc_fences='      float* od0 = &o[0][0][0];\n#pragma unroll\n      for (int e = 0; e < 32; ++e) reg_fence(od0[e]);\n#pragma unroll\n      for (int e = 0; e < 4; ++e) reg_fence(lacc[e]);'
assert prefix.count(acc_fences)==1
prefix=prefix.replace(acc_fences,'')
qk_wait='    wgmma_wait<0>();\n#pragma unroll\n    for (int e = 0; e < 32; ++e) reg_fence(s[e]);'
assert prefix.count(qk_wait)==1
prefix=prefix.replace(qk_wait,qk_wait+'\n'+acc_fences)
names=['&attn_kernel_fp8pv<8, true, 0>']
nv=NVRTC(); rc,log,cubin,seconds=nv.compile(prefix.encode(),names,flags,True)
out=Path('.tools/nvrtc-preflight/attention-l8-flat-delayed-fence')
out.with_suffix('.cu').write_bytes(prefix.encode());out.with_suffix('.log').write_text(log,encoding='utf-8')
if rc==0: out.with_suffix('.cubin').write_bytes(cubin)
report={'retained_source_sha256':hashlib.sha256(raw).hexdigest(),'cuda_source_sha256':hashlib.sha256(src.encode()).hexdigest(),'prefix_sha256':hashlib.sha256(prefix.encode()).hexdigest(),'removed_host_includes':removed,'host_launcher_omitted_at':anchor,'virtual_type_traits_header':TAGS,'virtual_stdint_header':STDINT,'flags':flags,'names':names,'return_code':rc,'cubin_bytes':len(cubin),'compile_seconds':seconds,'diagnostic_change':'Omit end drain and move output-accumulator empty fences from partial wait to immediately after full QK wait; retain P-register partial-wait fence and final drain. Not a submitted candidate, correctness unknown.','parent_prefix_sha256':hashlib.sha256(original_prefix.encode()).hexdigest(),'official_gpu_runs':0,'kernel_execution':False,'proxy_not_judge_NVCC':True}
out.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:report[k] for k in ['retained_source_sha256','return_code','cubin_bytes','compile_seconds','official_gpu_runs']}))
if rc: print(log[-5000:]);sys.exit(1)
