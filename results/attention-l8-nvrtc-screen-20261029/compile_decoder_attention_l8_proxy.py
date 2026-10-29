from pathlib import Path
import ast,hashlib,json,sys
sys.path.insert(0,str(Path('experiments/prepared').resolve()))
from local_nvrtc_compile import NVRTC,TAGS,STDINT,TOOL,PINNED
for name,h in PINNED.items(): assert hashlib.sha256((TOOL/'wheels'/name).read_bytes()).hexdigest()==h
raw=Path('best/decoder_verified.py').read_bytes()
assert hashlib.sha256(raw).hexdigest()=='d9fe7558624a1f070bbacb493b94302e90ff9bf77cdd8d4239f9d7a83ab39988'
tree=ast.parse(raw)
src=next(n.value.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_native_attn_CUDA_SRC' for t in n.targets))
anchor='template <int L>\nvoid launch('
assert src.count(anchor)==1
prefix=src.split(anchor)[0]
removed=['#include <torch/extension.h>\n','#include <ATen/cuda/CUDAContext.h>\n','#include <cuda.h>\n']
for s in removed:
 assert prefix.count(s)==1
 prefix=prefix.replace(s,'')
flags=['-DINFINITY=__int_as_float(0x7f800000)','--gpu-architecture=sm_90a','--std=c++20','--generate-line-info','--device-as-default-execution-space','--ptxas-options=-v','--include-path='+str(TOOL/'nvidia/cu13/include'),'--include-path='+str(TOOL/'nvidia/cu13/include/cccl')]
names=['&attn_kernel<8, true, 0>']
nv=NVRTC(); rc,log,cubin,seconds=nv.compile(prefix.encode(),names,flags,True)
out=Path('.tools/nvrtc-preflight/decoder-attention-l8-flat')
out.with_suffix('.cu').write_bytes(prefix.encode());out.with_suffix('.log').write_text(log,encoding='utf-8')
if rc==0: out.with_suffix('.cubin').write_bytes(cubin)
report={'retained_source_sha256':hashlib.sha256(raw).hexdigest(),'cuda_source_sha256':hashlib.sha256(src.encode()).hexdigest(),'prefix_sha256':hashlib.sha256(prefix.encode()).hexdigest(),'removed_host_includes':removed,'host_launcher_omitted_at':anchor,'virtual_type_traits_header':TAGS,'virtual_stdint_header':STDINT,'flags':flags,'names':names,'return_code':rc,'cubin_bytes':len(cubin),'compile_seconds':seconds,'official_gpu_runs':0,'kernel_execution':False,'proxy_not_judge_NVCC':True}
out.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:report[k] for k in ['retained_source_sha256','return_code','cubin_bytes','compile_seconds','official_gpu_runs']}))
if rc: print(log[-5000:]);sys.exit(1)
