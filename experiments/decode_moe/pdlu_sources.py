"""Private PDL twins for the actual retained U16/U32 unpooler paths."""

import ast
import hashlib
import json

from pdl16_sources import BASE_SHA, HERE, ROOT, Derivation, literal

OUT = ROOT / 'results/decode-moe-plan-20261030/pdlu'
U16_INST = (1, 2, 1, 2, 0, 0, 256, 0, 8192, 4096, 2, 4, 16, 1, 2)
U32_INST = (32, 64, 2, 2, 2, True, 2)


def const(node):
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return const(node.left) + const(node.right)
    return literal(node)


def parents():
    raw = (HERE / 'decoder_pdli.py').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == BASE_SHA
    tree = ast.parse(raw)
    values = {n.targets[0].id: n.value for n in tree.body
              if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)}
    # The retained later subscript mutations, not the earlier default dicts,
    # define these two selected geometries. Assert their exact pinned values.
    changes = {ast.unparse(n.targets[0]): ast.literal_eval(n.value) for n in tree.body
               if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Subscript)
               and ast.unparse(n.targets[0]) in ["dd_decodediet_CPSSM_TABLE_MT['unpooler', 2]",
                                                "dd_decodediet_tt_mw_CONFIGS[32, 'unpooler']"]}
    assert changes["dd_decodediet_CPSSM_TABLE_MT['unpooler', 2]"] == ((1, 2, 1, 2, 0, 0, 256), (2, 4, 16, 1))
    assert changes["dd_decodediet_tt_mw_CONFIGS[32, 'unpooler']"] == (U32_INST, 2)
    template = literal(values['dd_decodediet_CPSSM_CUDA_TEMPLATE'])
    args = ','.join(map(str, U16_INST))
    cuda16 = template.replace('@@CASES@@', 'case 0: return dd_cap2_b_cpssm_launch_t<' + args +
        '>(wp,x,out,esc,m,N,K,WR,WK,KU,S,t01,u01,stream,query);')
    names = ['dd_decodediet_tt_mw_CUDA_PREFIX', 'dd_decodediet_tt_mw_CUDA_CORE',
             'dd_decodediet_tt_mw_CUDA_MAP', 'dd_decodediet_tt_mw_CUDA_HOST']
    ns = {name: const(values[name]) for name in names}
    for name in ['dd_decodediet_tt_mw_mma', 'dd_decodediet_tt_mw_cuda_source']:
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        exec(compile(ast.Module(body=[fn], type_ignores=[]), 'pinned-pure-unpool-source', 'exec'), ns)
    cuda32 = ns['dd_decodediet_tt_mw_cuda_source']([U32_INST])
    return {'u16': cuda16, 'u32': cuda32}, {
        'u16': literal(values['dd_decodediet_CPSSM_CPP']),
        'u32': literal(values['dd_decodediet_tt_mw_CPP']) + '\nint pdlu32_occupancy();\n'}


def u16_twin(parent):
    d = Derivation(parent)
    d.edit('ld.global.nc.v4.u32', 'ld.global.cg.v4.u32')
    helper = '''__device__ __noinline__ void pdlu16_wait(uint32_t checksum) {
  asm volatile("{ .reg .b32 used; mov.b32 used, %0; griddepcontrol.wait; }" :: "r"(checksum) : "memory");
}
'''
    anchor = '// Never taken, but ptxas cannot prove it:'
    d.edit(anchor, helper + '\n' + anchor)
    anchor = '    uint32_t U0, uint32_t U1) {\n  const int m = CM ? CM : m_arg;'
    d.edit(anchor, '    uint32_t U0, uint32_t U1) {\n  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");\n  const int m = CM ? CM : m_arg;')
    anchor = '    for (int q=0; q<CP; ++q) stage_weights(q,q);'
    d.edit(anchor, '''    static_assert(FMT==1 && MT==2 && RT==1 && ST==2 && RED==0 && DIAG==0 && CP==2, "fixed CP2 U16 preamble");
''' + anchor + '\n    pdlu16_wait(my_e);')
    d.edit('cudaLaunchAttribute at[2];', 'cudaLaunchAttribute at[3];')
    anchor = '  if (query == 1) {'
    d.edit(anchor, '''  if (query == 0) {
    const int dep_attr=lc.numAttrs;
    at[dep_attr].id=cudaLaunchAttributeProgrammaticStreamSerialization;
    at[dep_attr].val.programmaticStreamSerializationAllowed=1;
    lc.attrs=at;lc.numAttrs=dep_attr+1;
  }
''' + anchor)
    d.edit('dd_cap2_b_cpssm_gemv', 'pdlu16_gemv', count=d.text.count('dd_cap2_b_cpssm_gemv'))
    d.edit('  cudaLaunchKernelEx(&lc, kern,', '  const cudaError_t pdlu_rc=cudaLaunchKernelEx(&lc, kern,')
    anchor = '  return cudaGetLastError() == cudaSuccess ? 0 : -8;'
    d.edit(anchor, '  if(pdlu_rc!=cudaSuccess)return -static_cast<int>(pdlu_rc);\n' + anchor)
    return d.finish()


def u32_twin(parent):
    d = Derivation(parent)
    helper = '''__device__ __noinline__ void pdlu32_wait(uint32_t checksum) {
  asm volatile("{ .reg .b32 used; mov.b32 used, %0; griddepcontrol.wait; }" :: "r"(checksum) : "memory");
}
'''
    anchor = 'template<int BN,int R,int WK,int PF,int Q,bool C12,int CAP>\n__global__'
    d.edit(anchor, helper + '\n' + anchor)
    anchor = '    const uint32_t* __restrict__ esc,__nv_bfloat16* out,int m,int n,int k,int split,uint32_t t0,uint32_t t1) {\n'
    d.edit(anchor, anchor + '''  static_assert(BN==32 && R==64 && WK==2 && PF==2 && Q==2 && C12 && CAP==2, "fixed U32 cluster twin");
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
''')
    # Remove only the initial activation issue; the lambda and every refill
    # remain exactly original. All input copies are reissued below the wait.
    initial = '  #pragma unroll\n  for(int s=0;s<Q;s++)loadx(s);'
    d.edit(initial, '  // PDL: initial activation TMA follows the independent weight preamble.')
    anchor = '  for(int p=0;p<PF;p++)load(p,p);'
    refs = ' ^ '.join('raw[%d][%d].%s' % (p, l, c) for p in range(2) for l in range(3) for c in 'xyzw')
    d.edit(anchor, anchor + '\n  pdlu32_wait(e0 ^ e1 ^ ' + refs + ');\n' + initial)
    old = '  cudaLaunchAttribute attr={};attr.id=cudaLaunchAttributeClusterDimension;attr.val.clusterDim={unsigned(split),1,1};\n  if(split>1){cfg.attrs=&attr;cfg.numAttrs=1;}'
    new = '''  cudaLaunchAttribute attr[2]={};
  int na=0;
  if(split>1){attr[na].id=cudaLaunchAttributeClusterDimension;attr[na].val.clusterDim={unsigned(split),1,1};++na;}
  attr[na].id=cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[na].val.programmaticStreamSerializationAllowed=1;++na;
  cfg.attrs=attr;cfg.numAttrs=na;'''
    d.edit(old, new)
    anchor = 'int info(int id){'
    d.edit(anchor, '''int pdlu32_occupancy(){
  auto kern=m32w_gemm<32,64,2,2,2,true,2>;
  cudaError_t rc=cudaFuncSetAttribute(kern,cudaFuncAttributeMaxDynamicSharedMemorySize,18432);
  if(rc!=cudaSuccess)return -static_cast<int>(rc);
  rc=cudaFuncSetAttribute(kern,cudaFuncAttributeNonPortableClusterSizeAllowed,1);
  if(rc!=cudaSuccess)return -static_cast<int>(rc);
  cudaLaunchConfig_t cfg={};cfg.gridDim=dim3(256,1,1);cfg.blockDim=dim3(256,1,1);cfg.dynamicSmemBytes=18432;
  cudaLaunchAttribute attr={};attr.id=cudaLaunchAttributeClusterDimension;attr.val.clusterDim={2,1,1};
  cfg.attrs=&attr;cfg.numAttrs=1;
  int clusters=0;rc=cudaOccupancyMaxActiveClusters(&clusters,kern,&cfg);
  return rc==cudaSuccess?clusters:-static_cast<int>(rc);
}
''' + anchor)
    d.edit('m32w_gemm', 'pdlu32_gemm', count=d.text.count('m32w_gemm'))
    return d.finish()


def derive():
    src, cpp = parents()
    OUT.mkdir(exist_ok=True)
    out = {}
    for name, make in [('u16', u16_twin), ('u32', u32_twin)]:
        modified, proof = make(src[name])
        proof.update(parent_sha256=hashlib.sha256(src[name].encode()).hexdigest(),
                     candidate_sha256=hashlib.sha256(modified.encode()).hexdigest())
        (OUT / (name + '-parent.cu')).write_bytes(src[name].encode())
        (OUT / (name + '-candidate.cu')).write_bytes(modified.encode())
        (OUT / (name + '.cpp')).write_bytes(cpp[name].encode())
        out[name] = (src[name], modified, proof)
    (OUT / 'binding.json').write_text(json.dumps(dict(parent_decode_sha256=BASE_SHA, GPU_jobs=0,
        sources={k: v[2] for k, v in out.items()}), indent=2) + '\n')
    return out, cpp


if __name__ == '__main__':
    for name, (old, new, proof) in derive()[0].items():
        print(name, len(old), len(new), proof['candidate_sha256'], proof['exact_inverse'])
