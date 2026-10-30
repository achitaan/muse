"""Compare generic versus explicit shared atomic lowering; no GPU calls."""
import ast,hashlib,json,re,subprocess
from pathlib import Path
from audit_attention_finoutline_cpu import Capture
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/decode-moe-plan-20261030/norm-group'

def main():
    t=ast.parse((ROOT/'best/moe_verified.py').read_bytes())
    defs=next(ast.literal_eval(n.value)for n in t.body if isinstance(n,ast.Assign)and getattr(n.targets[0],'id','')=='TINY_DEFINES')
    flags=list(defs)+['-DOKC_PRMTSC=1','-DOKC_DQ7=1','-DOKC_NS=3','-DOKC_NSF=3']
    results={}
    for label,path in [('generic',OUT/'pre-shared-address-e5c0/packaged.cu'),('shared',OUT/'rejected-shared-address-33a3/packaged.cu')]:
        source=path.read_bytes();c=Capture();r=c.compile(source.split(b'#include <torch/extension.h>',1)[0],['&norm_group::moe_group_fused<false>'],flags,tags=True,lineinfo=False)
        assert r['rc']==0,r['log'];binary=OUT/(label+'-codegen.cubin');binary.write_bytes(c.cubin)
        exe=ROOT.parents[2]/'.tools/cuda-compiler-13.0/nvidia/cu13/bin/cuobjdump.exe'
        sass=subprocess.check_output([str(exe),'-sass',str(binary)],text=True);(OUT/(label+'-codegen.sass')).write_bytes(sass.encode())
        lines=[s.strip()for s in sass.splitlines()if 'ATOM.E.ADD.F32'in s]
        results[label]=dict(CUDA_sha256=hashlib.sha256(source).hexdigest(),global_F32_atomic_sites=len(lines),address_space_query_sites=sass.count('QSPC.'),global_F32_instructions=lines,cubin_sha256=r['cubin_sha256'])
    benefit=results['shared']['address_space_query_sites']<results['generic']['address_space_query_sites']
    (OUT/'explicit-shared-codegen.json').write_bytes((json.dumps(dict(results=results,benefit_demonstrated=benefit,decision='CPU reject: no reduction of generic address checks'if not benefit else 'Further review required',GPU_calls=0,limitation='Static instruction sites,not dynamic execution counts or latency.'),indent=2)+'\n').encode());print(json.dumps({k:{x:v[x]for x in ['global_F32_atomic_sites','address_space_query_sites']}for k,v in results.items()},indent=2))

if __name__=='__main__':main()
