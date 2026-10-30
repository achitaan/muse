"""Bounded CPU compiler screen; emits CUBIN and SASS without loading CUDA."""
import argparse
import ast
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zlib

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
MAIN=ROOT.parents[2]
sys.path.insert(0,str(MAIN/'.slim/worktrees/opus-fresh/experiments/opus_fresh'))
import nvrtc_screen_alternatives as nv
from pdl_source import twin
from check_r2_cpu import device_part


class Capture(nv.NVRTC):
    def output(self,prog,kind):
        value=super().output(prog,kind)
        if kind=='CUBIN':self.cubin=value
        return value


def specs():
    raw=(HERE/'decoder_r2f.py').read_bytes()
    assert hashlib.sha256(raw).hexdigest().startswith('75fa2a73')
    values={}
    for n in ast.parse(raw).body:
        if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name):
            values.setdefault(n.targets[0].id,[]).append(n.value)
    def value(k):
        n=values[k][-1]
        if isinstance(n,ast.DictComp):
            assert k=='tt_v10_sm2_SMALLM2_ROUTES'
            assert ast.unparse(n)=="{key: value for (key, value) in tt_v10_sm2_SMALLM2_ROUTES.items() if key == ('pooler', 8)}"
            return {('pooler',8):ast.literal_eval(values[k][0])['pooler',8]}
        if isinstance(n,ast.Call) and getattr(n.func,'id','')=='_zsrc':return zlib.decompress(base64.b64decode(ast.literal_eval(n.args[0]))).decode()
        return ast.literal_eval(n)
    return [
      ('o1','smallm2',value('tt_v10_sm2_SMALLM2_CUDA_TEMPLATE'),value('V12_SM2_O1')['o',1][0],False),
      ('o8','v18s',value('V18S_CUDA_TEMPLATE'),value('V18_SMALL')['o',8][0],False),
      ('r1','v18s',(HERE/'r2f_v18s.cu').read_text(),tuple(value('V18_SMALL')['pooler',1][0][:-1])+(value('V18_SMALL')['pooler',1][0][-1]|64,),True),
      ('r8','smallm2',(HERE/'r2f_smallm2.cu').read_text(),tuple(value('tt_v10_sm2_SMALLM2_ROUTES')['pooler',8][0][:-1])+(value('tt_v10_sm2_SMALLM2_ROUTES')['pooler',8][0][-1]|64,),True),
    ]


def main():
    p=argparse.ArgumentParser();p.add_argument('--form',choices=('inline','call'),default='inline');p.add_argument('--only',default='o1');p.add_argument('--baseline',action='store_true');args=p.parse_args()
    out=ROOT/'results/decode-moe-plan-20261030/pdl-cpu'/('parent' if args.baseline else args.form)
    out.mkdir(parents=True,exist_ok=True)
    nvr=Capture();exe=MAIN/'.tools/cuda-compiler-13.0/nvidia/cu13/bin/cuobjdump.exe'
    records=[]
    for name,prefix,cuda,inst,fused in specs():
        if name not in args.only.split(','):continue
        cuda=cuda.replace('@@CASES@@','')
        source,binding=(cuda,{'parent_only':True}) if args.baseline else twin(cuda,prefix,inst,fused=fused,form=args.form)
        (out/(name+'.cu')).write_bytes(source.encode())
        (out/(name+'-binding.json')).write_bytes((json.dumps(binding,indent=2)+'\n').encode())
        proxy=device_part(source)
        expression=f'&{prefix}_gemv<'+','.join(map(str,inst))+'>'
        r=nvr.compile(proxy.encode(),[expression],[],tags=True,lineinfo=True)
        (out/(name+'-compile.log')).write_text(r['log'])
        r['kernels']=nv.parse_ptxas(r['log']) if r['rc']==0 else []
        r.update(name=name,source_sha256=hashlib.sha256(source.encode()).hexdigest(),proxy_sha256=hashlib.sha256(proxy.encode()).hexdigest(),CPU_only=True,GPU_jobs=0)
        (out/(name+'-compile.json')).write_bytes((json.dumps(r,indent=2)+'\n').encode())
        if r['rc']==0:
            cubin=out/(name+'.cubin');cubin.write_bytes(nvr.cubin)
            sass=subprocess.check_output([str(exe),'-sass',str(cubin)],text=True,encoding='utf-8')
            (out/(name+'.sass')).write_bytes(sass.encode())
            lines=[l.strip() for l in sass.splitlines() if any(x in l for x in ('Function :','ACQBULK','LDG.','CALL.'))]
            r['ordering_excerpt']=lines[:70]
        print(json.dumps({k:v for k,v in r.items() if k in ('name','rc','kernels','ordering_excerpt')},indent=2))
        if r['rc']:print(r['log'][-3500:])
        records.append(r)
    records=[json.loads(f.read_text()) for f in sorted(out.glob('*-compile.json'))]
    (out/'summary.json').write_bytes((json.dumps(records,indent=2)+'\n').encode())


if __name__=='__main__':main()
