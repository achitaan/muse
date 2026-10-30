"""Source-preserving M32 sliding integration of the two qualified handoffs."""

import ast
import base64
import hashlib
import json
import zlib

from pdl16_sources import HERE, ROOT, BASE_SHA
from build_norm_pair_diag import extract

OUT = ROOT / 'results/decode-moe-plan-20261030/pdl32-handoff'


def scoped(source, namespace, anchor):
    assert source.count(anchor) == 1
    opening = 'namespace ' + namespace + ' {\n'
    result = source.replace(anchor, opening + anchor, 1) + '\n} // ' + namespace + '\n'
    assert result.replace(opening, '', 1).removesuffix('\n} // ' + namespace + '\n') == source
    return result


HOST = r'''
int d32o_launch(int id,int64_t x,int64_t w,int64_t out,int m,int64_t stream,int query) {
  return d32o_private::ol_launch(id,x,w,out,m,stream,query);
}
int d32o_attr(int what) {
  auto fn=d32o_private::pdlo32_projection<32,128,8,1,1,1,1>;
  cudaFuncAttributes a{};cudaError_t rc=cudaFuncGetAttributes(&a,fn);
  if(rc!=cudaSuccess)return -1000-int(rc);
  if(what==0)return a.numRegs;
  if(what==1)return int(a.localSizeBytes);
  if(what==2)return int(a.sharedSizeBytes);
  if(what==3)return a.maxDynamicSharedSizeBytes;
  if(what==4)return a.maxThreadsPerBlock;
  if(what==5)return d32o_private::pdlo32_info(1);
  return -99;
}
int d32u_launch(int id,uint64_t x,uint64_t w,uint64_t e,uint64_t out,int m,int n,int k,int split,uint64_t tab,uint64_t stream) {
  return d32u_private::launch(id,x,w,e,out,m,n,k,split,tab,stream);
}
int d32u_info(int id) { return d32u_private::info(id); }
int d32u_occupancy() { return d32u_private::pdlu32_occupancy(); }
'''

CPP = '''#include <pybind11/pybind11.h>
#include <stdint.h>
int d32o_launch(int,int64_t,int64_t,int64_t,int,int64_t,int);
int d32o_attr(int);
int d32u_launch(int,uint64_t,uint64_t,uint64_t,uint64_t,int,int,int,int,uint64_t,uint64_t);
int d32u_info(int);
int d32u_occupancy();
'''


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    parent = (HERE / 'decoder_pdli.py').read_bytes()
    assert hashlib.sha256(parent).hexdigest() == BASE_SHA
    o = (OUT.parent / 'pdl16/o32-candidate.cu').read_bytes().decode()
    u = (OUT.parent / 'pdlu/u32-candidate.cu').read_bytes().decode()
    assert hashlib.sha256(o.encode()).hexdigest() == 'b45d6e23af85adc9cb16e41e5c02f445714188dd28d0fa5c8a14855e96526f7f'
    assert hashlib.sha256(u.encode()).hexdigest() == '4dea6ac736ea57a9c36f5e9757cc1505ed0eb7e3fe9dbb7bb5b1d315004cf63a'
    tested = (OUT.parent / 'handoff16-diag/projection-original-prefix.cu').read_bytes()
    assert hashlib.sha256(tested).hexdigest() == '09b371d0ffce2e93655e5ebdc188a5276e62c2217d62b63a3b8f139be49be220'
    assert o.encode() in tested and u.encode() in tested
    assert 'griddepcontrol.wait;' in o and 'griddepcontrol.wait;' in u
    cuda = scoped(o, 'd32o_private', '#define OD __device__ __forceinline__')
    cuda += scoped(u, 'd32u_private', '#define D __device__ __forceinline__') + HOST
    assert '#include' not in cuda.split('namespace d32o_private {',1)[1].split('} // d32o_private',1)[0]
    assert '#include' not in cuda.split('namespace d32u_private {',1)[1].split('} // d32u_private',1)[0]
    app = '\n# ---- isolated M32 handoffs; all other paths retain c15df9d3 ----\n'
    app += 'def _d32_zsrc(s):\n import base64,zlib\n return zlib.decompress(base64.b64decode(s)).decode()\n'
    app += '_D32_CUDA = _d32_zsrc(' + repr(base64.b64encode(zlib.compress(cuda.encode(),9)).decode()) + ')\n'
    for key,value in {'_D32_CUDA_SHA':hashlib.sha256(cuda.encode()).hexdigest(), '_D32_CPP':CPP,
        '_D32_FUNCTIONS':['d32o_launch','d32o_attr','d32u_launch','d32u_info','d32u_occupancy']}.items():
        app += key + ' = ' + repr(value) + '\n'
    chain = (HERE / 'decoder_pdl_chain_draft.py').read_bytes()
    assert hashlib.sha256(chain).hexdigest() == '985abda3f1a91e6daca07a38e9f91a397b4942ef892285704ddbd1792bff9a40'
    app += extract(chain.decode(), ['_p16_data']).replace('def _p16_data(', 'def _d32_data(',1)
    app += (HERE / 'pdl32_handoff_runtime.py').read_text(encoding='utf-8')
    raw = parent + app.encode()
    ast.parse(raw)
    assert raw.startswith(parent) and len(raw)<1_000_000
    (HERE / 'decoder_pdl32_handoff_draft.py').write_bytes(raw)
    (OUT / 'isolated.cu').write_bytes(cuda.encode());(OUT / 'isolated.cpp').write_bytes(CPP.encode())
    report = dict(source_sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),parent_sha256=BASE_SHA,
        parent_prefix_exact=True,CUDA_sha256=hashlib.sha256(cuda.encode()).hexdigest(),
        O32_source_sha256=hashlib.sha256(o.encode()).hexdigest(),U32_source_sha256=hashlib.sha256(u.encode()).hexdigest(),
        CUDA_derivation='Exact sources enclosed in unique namespaces after headers; inverse exact. New global host aliases and resource query only.',
        qualification_jobs={'O32':'ec1ea26451494f02','U32':'3264689ee9974ef9'},
        fixed_scope='M32, B32,L1,sliding only; both O and unpooler as one bundle. No component sweep.',
        unchanged='All expert/router/attention/norm arithmetic, PDL flags, weights, BF16 boundaries, scratch and other-regime dispatch.',
        gates='Actual H200 resources, both leaf bytes, complete layer/routing/cache bytes on changing inputs, >3% complete-layer cold-L2 graphs in both orders.',
        GPU_jobs=0,admitted_for_GPU=False,publication=False)
    (OUT / 'source-binding.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
