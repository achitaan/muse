"""One fixed real Attention/O32 chain with causal and residency controls."""

import ast
import base64
import hashlib
import json
import re
import zlib

from pdl16_sources import HERE, ROOT, BASE_SHA
from build_norm_pair_diag import extract, DENSE_SHA
from build_pdl32_handoff import scoped
from harness import append_harness

OUT=ROOT/'results/decode-moe-plan-20261030/real-attention-handoff'


def early_attention(original):
    old='  pdl_launch_dependents();'
    assert original.count(old)==2
    new=original.replace(old,'  /* outgoing signal moved to uniform kernel entry */')
    anchor='__global__ void __launch_bounds__(Cfg<L, FLAT>::NTX, Cfg<L, FLAT>::MINB) attn_kernel_fp8pv(const __grid_constant__ Params p) {'
    assert new.count(anchor)==1
    new=new.replace(anchor,anchor+'\n  pdl_launch_dependents();',1)
    inverse=new.replace(anchor+'\n  pdl_launch_dependents();',anchor,1)
    inverse=inverse.replace('  /* outgoing signal moved to uniform kernel entry */',old)
    assert inverse==original
    return new


def projection_host():
    return r'''
int eac_o8(int64_t x,int64_t w,int64_t out,int64_t stream,int query) {
 return eac_o8_private::ol_launch(0,x,w,out,32,stream,query);
}
int eac_o4(int64_t x,int64_t w,int64_t out,int64_t stream,int query) {
 return eac_o4_private::ol_launch(0,x,w,out,32,stream,query);
}
template<class Fn> int eac_attrs(Fn fn,int what,int shared) {
 cudaFuncAttributes a{};cudaError_t rc=cudaFuncGetAttributes(&a,fn);
 if(rc!=cudaSuccess)return -1000-int(rc);
 if(what==0)return a.numRegs;
 if(what==1)return int(a.localSizeBytes);
 if(what==2)return int(a.sharedSizeBytes);
 if(what==3)return a.maxDynamicSharedSizeBytes;
 if(what==4){int blocks=0;rc=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,fn,128,shared);return rc==cudaSuccess?blocks:-int(rc);}
 return -99;
}
int eac_oattr(int stage,int what) {
 if(stage==8)return eac_attrs(eac_o8_private::pdlo32_projection<32,128,8,1,1,1,1>,what,196736);
 if(stage==4)return eac_attrs(eac_o4_private::pdlo32_projection<32,128,4,1,1,1,1>,what,98368);
 return -98;
}
'''


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    dense=(ROOT/'best/verified.py').read_bytes();decode=(HERE/'decoder_pdli.py').read_bytes()
    assert hashlib.sha256(dense).hexdigest()==DENSE_SHA and hashlib.sha256(decode).hexdigest()==BASE_SHA
    text=decode.decode();lines=text.splitlines(keepends=True);tree=ast.parse(text)
    def named(n):
        if isinstance(n,(ast.FunctionDef,ast.ClassDef)):return n.name
        if isinstance(n,ast.Assign)and isinstance(n.targets[0],ast.Name):return n.targets[0].id
        return ''
    nodes=[n for n in tree.body if named(n).startswith('_native_attn_')]
    native=''.join(''.join(lines[n.lineno-1:n.end_lineno])+'\n' for n in nodes)
    ast.parse(native)
    values={named(n):n.value for n in nodes if isinstance(n,ast.Assign)}
    original=ast.literal_eval(values['_native_attn_CUDA_SRC']);cpp=ast.literal_eval(values['_native_attn_CPP_SRC'])
    assert hashlib.sha256(original.encode()).hexdigest()=='241b67fe0008bbae53435b53d1bb6d647ded0b5718118875c0ec368b718b41db'
    changed=early_attention(original)
    assert changed.count('namespace {')==1
    private=changed.replace('namespace {','namespace eac_attention_private {\nnamespace {',1)+'\n} // eac_attention_private\n'
    declaration=cpp.strip();assert declaration.startswith('void run(')and declaration.endswith(';')
    args=declaration[declaration.index('(')+1:declaration.rindex(')')]
    names=[re.search(r'([A-Za-z_][A-Za-z_0-9]*)\s*$',arg.strip())[1]for arg in args.split(',')]
    public_decl=declaration.replace('void run(','void eac_run(',1)
    private+='\n'+public_decl[:-1]+' { eac_attention_private::run('+','.join(names)+'); }\n'
    private+=r'''
int eac_info(int what) {
 auto fn=eac_attention_private::attn_kernel_fp8pv<1,false,1>;
 cudaFuncAttributes a{};cudaError_t rc=cudaFuncGetAttributes(&a,fn);
 if(rc!=cudaSuccess)return -1000-int(rc);
 if(what==0)return a.numRegs;
 if(what==1)return int(a.localSizeBytes);
 if(what==2)return int(a.sharedSizeBytes);
 if(what==3)return a.maxDynamicSharedSizeBytes;
 if(what==4){int n=0;rc=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n,fn,128,58368);return rc==cudaSuccess?n:-int(rc);}
 return -99;
}
'''
    o8=(OUT.parent/'pdl16/o32-candidate.cu').read_text(encoding='utf-8')
    o4=(OUT.parent/'pdl32-residency/o32-st4.cu').read_text(encoding='utf-8')
    assert hashlib.sha256(o4.encode()).hexdigest()=='2a78e19708d641efa08f6cef14add5526c4af424f590ce18b274370431e1bfdd'
    projections=scoped(o8,'eac_o8_private','#define OD __device__ __forceinline__')
    projections+=scoped(o4,'eac_o4_private','#define OD __device__ __forceinline__')+projection_host()
    pc='#include <pybind11/pybind11.h>\n#include <stdint.h>\nint eac_o8(int64_t,int64_t,int64_t,int64_t,int);\nint eac_o4(int64_t,int64_t,int64_t,int64_t,int);\nint eac_oattr(int,int);\n'
    app=append_harness('')+'\nimport hashlib, math\nfrom typing import TypedDict\n'
    app+='def _eac_zsrc(s):\n import base64,zlib\n return zlib.decompress(base64.b64decode(s)).decode()\n'
    for key,value in {'_EAC_NATIVE_PY':native,'_EAC_EARLY_CUDA':private,'_EAC_O_CUDA':projections}.items():
        app+=key+' = _eac_zsrc('+repr(base64.b64encode(zlib.compress(value.encode(),9)).decode())+')\n'
    for key,value in {'_EAC_EARLY_CPP':public_decl+'\nint eac_info(int);\n','_EAC_O_CPP':pc,
        '_EAC_EARLY_SHA':hashlib.sha256(private.encode()).hexdigest(),
        '_EAC_O_SHA':hashlib.sha256(projections.encode()).hexdigest(),
        'ROPE_THETA':ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign)and named(n)=='ROPE_THETA'))}.items():
        app+=key+' = '+repr(value)+'\n'
    app+=extract(text,['_pdli_equal'])
    chain=(HERE/'decoder_pdl_chain_draft.py').read_text(encoding='utf-8')
    app+=extract(chain,['_p16_data']).replace('def _p16_data(','def _eac_data(',1)
    app+=(HERE/'real_attention_handoff_runtime.py').read_text(encoding='utf-8')
    raw=dense+app.encode();ast.parse(raw);assert len(raw)<1_000_000 and raw.startswith(dense)
    (HERE/'dense_real_attention_handoff.py').write_bytes(raw)
    for name,value in [('native.py',native),('attention-parent.cu',original),('attention-early-unscoped.cu',changed),
        ('attention-private.cu',private),('attention-private.cpp',public_decl+'\nint eac_info(int);\n'),('projections.cu',projections),('projections.cpp',pc)]:
        (OUT/name).write_bytes(value.encode())
    record=dict(source_sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),Dense_parent_sha256=DENSE_SHA,
        Decode_parent_sha256=BASE_SHA,native_Python_sha256=hashlib.sha256(native.encode()).hexdigest(),
        original_Attention_CUDA_sha256=hashlib.sha256(original.encode()).hexdigest(),
        early_Attention_CUDA_sha256=hashlib.sha256(changed.encode()).hexdigest(),
        private_Attention_CUDA_sha256=hashlib.sha256(private.encode()).hexdigest(),
        projection_CUDA_sha256=hashlib.sha256(projections.encode()).hexdigest(),
        fixed_scope='M32/B32/L1/sliding real Attention then O32, actual Dense weights; no MoE or whole-layer claim.',
        arms=['parent: retained Attention + original O32','late8: retained Attention + incoming-PDL ST8',
              'early8: entry-signaled Attention + incoming-PDL ST8','early4: entry-signaled Attention + incoming-PDL ST4'],
        precision='All Attention arithmetic and cache handling unchanged; only outgoing signal placement. ST4 preserves original K/MMA order, adds original ring refills.',
        gates='Finite exact intermediate/output/cache bytes on changing normal/zero/outlier inputs. >3% and >=1.5us both arm orders. Early8 additionally beats late8; early4 additionally beats early8.',
        GPU_jobs=0,enables_nothing=True,timed_Dense_parent_unchanged=True,admitted_for_GPU=False)
    (OUT/'source-binding.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8');print(json.dumps(record,indent=2))


if __name__=='__main__':main()
