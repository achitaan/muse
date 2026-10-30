"""CPU-only resource and SASS screen of the fixed integration kernels."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

from build_pdli import derive, RESULTS
from check_pdl_diag_cpu import Capture, MAIN
from check_r2_cpu import device_part, PRE
import nvrtc_screen_alternatives as nv


def main():
    p=argparse.ArgumentParser();p.add_argument('--only',default='o1,o8,r1,r8,u1,u8,q1,q8');p.add_argument('--packed',action='store_true');a=p.parse_args()
    _,sources,specs,_=derive();out=RESULTS/'pdli-cpu';nvr=Capture();summary={}
    if a.packed:
        proxy=PRE+'\n#include <cooperative_groups.h>\n'
        expressions=[]
        for name,src in sources.items():
            body=device_part(src)[len(PRE):]
            body=re.sub(r'^\s*#include\s+<[^>]+>[^\n]*\n','',body,flags=re.M)
            proxy+='\nnamespace pdli_'+name+' {\n'+body+'\n}\n#undef SMD\n'
            if '#define DC2_DECODE' in body:proxy+='#undef DC2_DECODE\n'
            spec=specs[name]
            expressions.append('&pdli_'+name+'::'+spec['prefix']+'_gemv<'+','.join(map(str,spec['instance']))+'>')
        (out/'combined-device-proxy.cu').write_bytes(proxy.encode())
        r=nvr.compile(proxy.encode(),expressions,[],tags=True,lineinfo=True)
        (out/'combined-compile.log').write_bytes(r['log'].encode())
        r['kernels']=nv.parse_ptxas(r['log']) if r['rc']==0 else []
        if r['rc']==0:
            cubin=out/'combined.cubin';cubin.write_bytes(nvr.cubin)
            sass=subprocess.check_output([str(MAIN/'.tools/cuda-compiler-13.0/nvidia/cu13/bin/cuobjdump.exe'),'-sass',str(cubin)],text=True)
            (out/'combined.sass').write_bytes(sass.encode())
        (out/'combined-compile.json').write_bytes((json.dumps(r,indent=2)+'\n').encode())
        print('packed rc',r['rc'],'kernels',r['kernels'])
        if r['rc']:print(r['log'][-4000:])
        return
    for name in a.only.split(','):
        src=sources[name];spec=specs[name];inst=spec['instance'];prefix=spec['prefix']
        proxy=device_part(src)
        r=nvr.compile(proxy.encode(),['&'+prefix+'_gemv<'+','.join(map(str,inst))+'>'],[],tags=True,lineinfo=True)
        (out/(name+'-compile.log')).write_bytes(r['log'].encode())
        r['source_sha256']=hashlib.sha256(src.encode()).hexdigest()
        r['kernels']=nv.parse_ptxas(r['log']) if r['rc']==0 else []
        if r['rc']==0:
            cubin=out/(name+'.cubin');cubin.write_bytes(nvr.cubin)
            sass=subprocess.check_output([str(MAIN/'.tools/cuda-compiler-13.0/nvidia/cu13/bin/cuobjdump.exe'),'-sass',str(cubin)],text=True)
            (out/(name+'.sass')).write_bytes(sass.encode())
            ins=[]
            for line in sass.splitlines():
                match=re.search(r'/\*([0-9a-f]+)\*/\s+(.+?);',line)
                if match:ins.append((int(match[1],16),match[2]))
            trigger=next((addr,text) for addr,text in ins if 'PREEXIT' in text)
            r['outgoing_trigger']=trigger
            if spec['kind'] in ('o','r','u'):
                call=next((addr,text) for addr,text in ins if 'CALL.REL.NOINC' in text)
                target=int(re.search(r'0x([0-9a-f]+)',call[1])[1],16)
                before=[(addr,text) for addr,text in ins if addr<call[0] and 'LDG.' in text]
                after=[(addr,text) for addr,text in ins if addr>call[0] and 'LDG.' in text and 'STRONG.GPU' in text]
                expected=4 if spec['kind']=='r' else 7
                assert len(before)==expected,(name,before)
                assert all('CONSTANT' in text for addr,text in before),(name,before)
                assert after,(name,'missing coherent dependent load')
                assert any(target<=addr<target+96 and 'ACQBULK' in text for addr,text in ins),(name,'bad wait target')
                r['ordering']={'wait_call':call,'helper_target':target,'independent_loads':before,'first_dependent_load':after[0]}
            print(name,r['kernels'],'trigger',trigger,'ordering',r.get('ordering'))
        else:
            print(name,r['rc'],r['log'][-3500:])
        (out/(name+'-compile.json')).write_bytes((json.dumps(r,indent=2)+'\n').encode())
        summary[name]=r
    (out/'compile-summary.json').write_bytes((json.dumps(summary,indent=2)+'\n').encode())


if __name__=='__main__':main()
