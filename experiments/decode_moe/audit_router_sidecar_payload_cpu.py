"""Packaged device identity, fixed ABI and public delegation checks without GPU access."""
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
from types import SimpleNamespace
from build_router_sidecar_payload import ROOT, OUT, DENSE, sources


def runtime_contract(raw):
    tree = ast.parse(raw)
    nodes = {n.name:n for n in tree.body if isinstance(n, ast.FunctionDef)}
    calls = []
    def parent(data, state):
        calls.append((data, state))
        return 'retained-result'
    ns = {'_rs_parent_kernel_fn':parent}
    exec(compile(ast.Module(body=[nodes['kernel_fn']], type_ignores=[]),'public-delegation','exec'), ns)
    for layer in ('qkv_gate','o','pooler','unpooler','router'):
        for m in (1,2,8,16,32,128,256,257):
            data, state = {'config':{'layer':layer,'m':m}}, {'opaque':object()}
            assert ns['kernel_fn'](data,state) == 'retained-result'
            assert calls[-1][0] is data and calls[-1][1] is state
    class Tensor:
        next_pointer = 256
        def __init__(self, shape, dtype=None, device=None):
            self.shape,self.dtype,self.device=shape,dtype,device
            self.pointer=Tensor.next_pointer
            Tensor.next_pointer+=256
        def data_ptr(self):return self.pointer
    outputs=[]
    def empty(shape, **kw):
        t=Tensor(shape,**kw);outputs.append(t);return t
    recorded=[]
    def launch(*args):recorded.append(args);return 0
    torch=SimpleNamespace(empty=empty,bfloat16='bf16',float32='f32',int32='i32',
                          cuda=SimpleNamespace(current_stream=lambda:SimpleNamespace(cuda_stream=123)))
    scope={'torch':torch}
    exec(compile(ast.Module(body=[nodes['_rs_run']], type_ignores=[]),'fixed-native-ABI','exec'),scope)
    plan={'wp':Tensor((1,)),'esc':Tensor((1,)),'cfg':(2,16,8,1),'t01':999,'u01':1001}
    state={k:Tensor((1,))for k in ('weight','bias','parts','counter')}
    xb,xr=Tensor((1,8192),device='gpu'),Tensor((1,8192),device='gpu')
    result=scope['_rs_run'](SimpleNamespace(r2_launch=launch),plan,xb,xr,state)
    args=recorded[-1];assert len(args)==22
    assert args[:2]==(plan['wp'].data_ptr(),xb.data_ptr())
    assert args[3:15]==(plan['esc'].data_ptr(),1,4096,8192,2,16,8,1,999,1001,123,xr.data_ptr())
    assert args[15:19]==tuple(state[k].data_ptr()for k in ('weight','bias','parts','counter'))
    assert args[-1]==0 and tuple(t.shape for t in result)==((1,4096),(1,8),(1,8))
    try:
        scope['_rs_run'](SimpleNamespace(r2_launch=lambda *a:-20),plan,xb,xr,state)
    except RuntimeError:
        pass
    else:
        raise AssertionError('failed native launch was silently accepted')
    return {'public_delegations':len(calls),'native_ABI_arguments':22,'native_failure_raises':True}


def main():
    raw=(ROOT/'experiments/decode_moe/dense_router_sidecar_diagnostic.py').read_bytes()
    binding=json.loads((OUT/'payload-binding.json').read_text())
    assert hashlib.sha256(raw).hexdigest()==binding['source_sha256']
    base=(ROOT/'best/verified.py').read_bytes()
    assert hashlib.sha256(base).hexdigest()==DENSE and raw.startswith(base.rstrip(b'\n'))
    contract=runtime_contract(raw)
    from check_r2_cpu import device_part
    from check_pdl_diag_cpu import Capture,MAIN
    import nvrtc_screen_alternatives as nv
    parent,candidate,cpp=sources()
    results={}
    for label,source,name in [('parent',parent,'v18s_gemv'),('candidate',candidate,'sidecar_gemv')]:
        assert (OUT/(label+'-packaged.cu')).read_bytes()==source.encode()
        compiler=Capture()
        r=compiler.compile(device_part(source).encode(),['&'+name+'<1,1,1,1,0,0,1024,0,4096,8192,2,16,8,1,70>'],[],tags=True,lineinfo=True)
        (OUT/(label+'-packaged.log')).write_bytes(r['log'].encode())
        assert r['rc']==0,r['log'][-3000:]
        binary=OUT/(label+'-packaged.cubin');binary.write_bytes(compiler.cubin)
        sass=subprocess.check_output([str(MAIN/'.tools/cuda-compiler-13.0/nvidia/cu13/bin/cuobjdump.exe'),'-sass',str(binary)],text=True)
        (OUT/(label+'-packaged.sass')).write_bytes(sass.encode())
        instructions=lambda s:re.findall(r'/\*([0-9a-f]+)\*/\s+(.+?);',s)
        assert instructions(sass)==instructions((OUT/(label+'.sass')).read_text())
        kernels=nv.parse_ptxas(r['log'])
        assert all(k['registers']==64 and k['stack_frame']==k['spill_stores']==k['spill_loads']==0 and k['smem_static']==32768 for k in kernels)
        results[label]={'CUDA_sha256':hashlib.sha256(source.encode()).hexdigest(),'cubin_sha256':r['cubin_sha256'],
                        'kernels':kernels,'all_instructions_match_screened_prototype':True}
        print(label,'packaged device unchanged except kernel name;64 regs,no spills',flush=True)
    report={'source_sha256':binding['source_sha256'],'runtime_contract':contract,'compiler':results,
            'CPP_declaration_sha256':hashlib.sha256(cpp.encode()).hexdigest(),
            'GPU_jobs':0,'admitted_for_GPU':False,
            'limitation':'Mock ABI/delegation checks and exact packaged device instructions. Host NVCC/linking,actual occupancy,graph replay,markers and performance still require the one admitted diagnostic.'}
    (OUT/'payload-audit.json').write_bytes((json.dumps(report,indent=2)+'\n').encode())
    print(json.dumps(contract))


if __name__=='__main__':main()
