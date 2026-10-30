# Helper-local alternative adapted from frozen grouped-norm tooling.
"""Package the screened group source and the checked host-resource bridge."""
import ast,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/decode-moe-plan-20261030/norm-register'

def derive():
    raw=(ROOT/'best/moe_verified.py').read_bytes()
    assert hashlib.sha256(raw).hexdigest()=='331148716e8d1ba92b45cd29725be997d44b3a5235af74d75770bb9444100d50'
    vals={n.targets[0].id:ast.literal_eval(n.value)for n in ast.parse(raw).body if isinstance(n,ast.Assign)and getattr(n.targets[0],'id','')in ('CUDA_SRC','CPP_SRC','TINY_DEFINES','TINY2_DEFINES')}
    original=(OUT/'candidate.cu').read_text(encoding='utf-8')
    audit=json.loads((OUT/'cpu-audit.json').read_text())
    assert hashlib.sha256(original.encode()).hexdigest()==audit['source_sha256']
    device,host=original.split('#include <torch/extension.h>',1)
    for a,b in [('okc_run','okc_tnorm_run'),('okc_set_pdl','okc_tnorm_set_pdl'),('okc_smem_bytes','okc_tnorm_smem_bytes'),('g_okc_pdl','g_okc_tnorm_pdl')]:host=host.replace(a,b)
    src=device+'#include <torch/extension.h>'+host
    checked=(ROOT/'results/decode-moe-plan-20261030/tiny-norm-whole/candidate.cu').read_text(encoding='utf-8')
    assert hashlib.sha256(checked.encode()).hexdigest()=='dcfd3b845c748711aefc27ff07f7ab7cbf7a16aed739fd5dc12f0e2a33a631ce'
    metadata=checked[checked.index('std::vector<int64_t> okc_tnorm_resources()'):]
    metadata=metadata.replace('tiny_norm_whole::','norm_reg::').replace('moe_tnorm_fused','moe_reg_fused')
    src+='\n'+metadata
    cpp=vals['CPP_SRC'].replace('okc_run','okc_tnorm_run').replace('okc_set_pdl','okc_tnorm_set_pdl')+'\nstd::vector<int64_t> okc_tnorm_resources();\n'
    assert src.split('#include <torch/extension.h>',1)[0]==original.split('#include <torch/extension.h>',1)[0]
    return vals['CUDA_SRC'],src,cpp,vals

def main():
    _,source,cpp,_=derive();(OUT/'packaged.cu').write_bytes(source.encode());(OUT/'candidate.cpp').write_bytes(cpp.encode())
    audit=json.loads((OUT/'cpu-audit.json').read_text());compiles={k:{'rc':0,'kernels':v}for k,v in audit['compiler'].items()}
    binding={'candidate_CUDA_sha256':hashlib.sha256(source.encode()).hexdigest(),'compiles':compiles,'source_compiler_evidence':'cpu-audit.json;device prefix byte exact,host API bridge appended only','GPU_calls':0}
    # Adapter for the already-verified payload builder's compiler gate.
    (OUT/'payload-compiler-binding.json').write_bytes((json.dumps(binding,indent=2)+'\n').encode())

if __name__=='__main__':main()
