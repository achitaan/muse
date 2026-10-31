"""Packaged M1 device and ABI/shape-selector audit; no GPU calls."""
import ast
import hashlib
import json
import re
import subprocess
from types import SimpleNamespace
from build_router_m1_payload import ROOT,OUT,DENSE,sources
from audit_router_sidecar_payload_cpu import runtime_contract


def main():
    raw=(ROOT/'experiments/decode_moe/dense_router_m1_diagnostic.py').read_bytes()
    binding=json.loads((OUT/'payload-binding.json').read_text());assert hashlib.sha256(raw).hexdigest()==binding['source_sha256']
    assert raw.startswith((ROOT/'best/verified.py').read_bytes().rstrip(b'\n'))
    contract=runtime_contract(raw.decode().replace('_rm_','_rs_').replace('RM_','RS_'))
    tree=ast.parse(raw);node=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=='_M1Extension')
    ns={};exec(compile(ast.Module(body=[node],type_ignores=[]),'shape-selector','exec'),ns)
    seen=[]
    def fn(label):
        return lambda *args:(seen.append((label,args))or label)
    adapter=ns['_M1Extension'](SimpleNamespace(r2_launch=fn('retained')),SimpleNamespace(r2_launch=fn('specialized')))
    for m in range(258):
        args=tuple(m if i==4 else i for i in range(22));expected='specialized'if m==1 else'retained'
        assert adapter.r2_launch(*args)==expected and seen[-1]==(expected,args)
    from check_r2_cpu import device_part
    from check_pdl_diag_cpu import Capture,MAIN
    import nvrtc_screen_alternatives as nv
    results={}
    parent,candidate,cpp=sources()
    for label,source,name,cm,regs,shared in [('parent',parent,'v18s_gemv',0,64,32768),('candidate',candidate,'m1_gemv',1,57,4096)]:
        assert(OUT/(label+'-packaged.cu')).read_bytes()==source.encode()
        compiler=Capture();expr='&'+name+'<1,1,1,1,0,0,1024,'+str(cm)+',4096,8192,2,16,8,1,70>'
        r=compiler.compile(device_part(source).encode(),[expr],[],tags=True,lineinfo=True)
        (OUT/(label+'-packaged.log')).write_bytes(r['log'].encode());assert r['rc']==0,r['log'][-3000:]
        cubin=OUT/(label+'-packaged.cubin');cubin.write_bytes(compiler.cubin)
        sass=subprocess.check_output([str(MAIN/'.tools/cuda-compiler-13.0/nvidia/cu13/bin/cuobjdump.exe'),'-sass',str(cubin)],text=True)
        (OUT/(label+'-packaged.sass')).write_bytes(sass.encode())
        ins=lambda s:re.findall(r'/\*([0-9a-f]+)\*/\s+(.+?);',s)
        assert ins(sass)==ins((OUT/(label+'.sass')).read_text())
        kernels=nv.parse_ptxas(r['log']);assert all(k['registers']==regs and k['smem_static']==shared and k['stack_frame']==k['spill_stores']==k['spill_loads']==0 for k in kernels)
        instructions=[(int(a,16),t.strip())for a,t in ins(sass)]
        call=next((a,t)for a,t in instructions if 'CALL.REL.NOINC'in t)
        target=int(re.search(r'0x([0-9a-f]+)',call[1])[1],16)
        independent=[(a,t)for a,t in instructions if a<call[0]and'LDG.'in t]
        assert len(independent)==4 and all('CONSTANT'in t for a,t in independent)
        assert any(target<=a<target+96 and'ACQBULK'in t for a,t in instructions)
        coherent=next((a,t)for a,t in instructions if a>call[0]and'LDG.'in t and'STRONG.GPU'in t)
        assert sum('PREEXIT'in t for a,t in instructions)==1
        assert sum('ACQBULK'in t for a,t in instructions)==2
        results[label]={'CUDA_sha256':hashlib.sha256(source.encode()).hexdigest(),'cubin_sha256':r['cubin_sha256'],
                        'kernels':kernels,'instructions_match_screened_prototype':True,
                        'PDL':{'wait_call':call,'wait_target':target,'independent_loads':independent,'first_coherent_load':coherent,
                               'outgoing_trigger_count':1,'wait_instruction_count':2}}
        print(label,'packaged instructions/resource/PDL screen passes',flush=True)
    record={'source_sha256':binding['source_sha256'],'contract':contract,'shape_selector_cases':len(seen),
            'compiler':results,'GPU_jobs':0,'admitted_for_GPU':False,
            'limitation':'CPU ABI/selector and fixed instruction/source checks;host build,GPU numerical/replay/PDL behavior and latency remain unproved.'}
    (OUT/'payload-audit.json').write_bytes((json.dumps(record,indent=2)+'\n').encode());print(json.dumps(contract),len(seen),'shape cases')


if __name__=='__main__':main()
