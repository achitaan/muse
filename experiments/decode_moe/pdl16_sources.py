"""Fixed O16/O32 early-start twins from the exact scored Decode source.

Pure source derivation only. No imports of submission code, GPU initialization,
dispatch changes or modification of the scored file.
"""

import ast
import base64
import hashlib
import json
from pathlib import Path
import zlib

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / 'results/decode-moe-plan-20261030/pdl16'
BASE_SHA = 'c15df9d3d83c55f4ba64be810e56502563e6cfe22bb5ef22172a1feb287654e5'
M16_INST = (1, 2, 1, 1, 0, 0, 256, 0, 8192, 1024, 2, 4, 4, 1, 7, 1)
O32_INST = (32, 128, 8, 1, 1, 1, 1)
M16_PREFIX = 'm16trace_59fcb5b72994'


def literal(node):
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == '_zsrc':
        return zlib.decompress(base64.b64decode(ast.literal_eval(node.args[0]))).decode()
    if isinstance(node, ast.Dict):
        return {literal(k): literal(v) for k, v in zip(node.keys, node.values)}
    if isinstance(node, (ast.Tuple, ast.List)):
        return tuple(literal(v) for v in node.elts)
    return ast.literal_eval(node)


def parents():
    raw = (HERE / 'decoder_pdli.py').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == BASE_SHA
    tree = ast.parse(raw)
    values = {n.targets[0].id: n.value for n in tree.body
              if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)}
    cpp16, cuda16, export16 = literal(values['mt_M16TRACE_SOURCES'])['o']
    route = literal(values['mt_M16TRACE_ROUTES'])['o']
    assert route['inst'] == M16_INST and route['cfg'] == (2, 4, 4, 1)
    assert route['variant'] == 'next_early' and export16 == M16_PREFIX + '_launch'
    ns = {'b2o_OLAYER_CUDA': literal(values['b2o_OLAYER_CUDA'])}
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'b2o_olayer_source')
    exec(compile(ast.Module(body=[fn], type_ignores=[]), 'pinned-pure-olayer-source', 'exec'), ns)
    cuda32 = ns['b2o_olayer_source']([O32_INST[:-1]])
    cpp32 = literal(values['b2o_OLAYER_CPP']).replace('r3_o32_o_direct_a_', '')
    return {'o16': cuda16, 'o32': cuda32}, {'o16': cpp16, 'o32': cpp32}


class Derivation:
    def __init__(self, parent):
        self.parent = self.text = parent
        self.edits = []

    def edit(self, old, new, count=1):
        assert self.text.count(old) == count, (old[:100], self.text.count(old), count)
        self.text = self.text.replace(old, new)
        self.edits.append((old, new, count))

    def finish(self):
        inverse = self.text
        for old, new, count in reversed(self.edits):
            # Deletions have a unique re-insertion anchor recorded as part of
            # the replaced line, rather than an empty replacement.
            assert new and inverse.count(new) == count, (new[:100], inverse.count(new), count)
            inverse = inverse.replace(new, old)
        assert inverse == self.parent
        return self.text, {'exact_inverse': True, 'edits': [dict(old=a, new=b, count=c) for a, b, c in self.edits]}


def o16_twin(parent):
    d = Derivation(parent)
    p = M16_PREFIX
    d.edit('ld.global.nc.v4.u32', 'ld.global.cg.v4.u32')
    helper = '''__device__ __noinline__ void pdlo16_wait(uint32_t checksum) {
  asm volatile("{ .reg .b32 used; mov.b32 used, %0; griddepcontrol.wait; }" :: "r"(checksum) : "memory");
}
'''
    anchor = 'template <int FMT, int MT, int RT, int ST, int RED, int DIAG, int NTMAX, int CM, int CN, int CK, int CWR, int CWK, int CKU, int CS, int XM, int TG>\n__global__'
    d.edit(anchor, helper + '\n' + anchor)
    entry = '    uint32_t U0, uint32_t U1) {\n  const int m = CM ? CM : m_arg;'
    d.edit(entry, '    uint32_t U0, uint32_t U1) {\n  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");\n  const int m = CM ? CM : m_arg;')
    refs = ' ^ '.join('(uint32_t)w[0][0][%d].%s' % (r, c) for r in range(3) for c in 'xyzw')
    old = '#pragma unroll\n    for (int s = 0; s < ST; ++s) load(s, s);'
    new = ('static_assert(FMT==1 && MT==2 && RT==1 && ST==1 && RED==0 && DIAG==0 && XM==7 && TG==1, "fixed O16 preamble");\n'
           + old + '\n    pdlo16_wait(my_e ^ ' + refs + ');')
    d.edit(old, new)
    bf = f'case 1: return {p}_launch_t<0,2,1,1,0,0,256,0,8192,1024,2,4,4,1,7,1>(wp,x,out,esc,m,N,K,WR,WK,KU,S,t01,u01,stream,query);'
    d.edit(bf, '// Private O16 twin supports only exact C12 plans; callers keep the original BF16 fallback.')
    d.edit('cudaLaunchAttribute at[1];', 'cudaLaunchAttribute at[2];')
    anchor = '  if (query == 1) {'
    d.edit(anchor, '''  if (query == 0) {
    const int pdli = lc.numAttrs;
    at[pdli].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at[pdli].val.programmaticStreamSerializationAllowed = 1;
    lc.attrs = at; lc.numAttrs = pdli + 1;
  }
''' + anchor)
    name = p + '_gemv'
    d.edit(name, 'pdlo16_gemv', count=d.text.count(name))
    d.edit('  cudaLaunchKernelEx(&lc, kern,', '  const cudaError_t pdli_launch_rc = cudaLaunchKernelEx(&lc, kern,')
    anchor = '  return cudaGetLastError() == cudaSuccess ? 0 : -8;'
    d.edit(anchor, '  if (pdli_launch_rc != cudaSuccess) return -static_cast<int>(pdli_launch_rc);\n' + anchor)
    return d.finish()


def o32_twin(parent):
    d = Derivation(parent)
    # Private namespace spelling only. The current direct-register store
    # epilogue and timed barrier watchdog remain byte-for-byte in the source.
    prefix = 'r3_o32_o_direct_a_ol_'
    d.edit(prefix, 'ol_', count=d.text.count(prefix))
    anchor = 'template<int BN,int BK,int ST,int PIPE,int PACK,int HINT,int TW>\n__global__'
    d.edit(anchor, '''__device__ __noinline__ void pdlo32_wait() {
  asm volatile("griddepcontrol.wait;" ::: "memory");
}
''' + anchor)
    entry = '    const uint8_t* w,__nv_bfloat16* out,int m) {\n'
    d.edit(entry, entry + '''  static_assert(BN==32 && BK==128 && ST==8 && PIPE==1 && PACK==1 && HINT==1 && TW==1, "fixed full-K O32 ring");
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
''')
    d.edit('  auto load=[&](int i) {', '  auto loadw=[&](int i) {')
    old = '    ol_tma(base+s*SB+AB,&tx,mt*T,i*(BK/64),full+s*8,px);\n  };'
    new = '''  };
  auto loadx=[&](int i) {
    const int s=i%ST;
    ol_tma(base+s*SB+AB,&tx,mt*T,i*(BK/64),full+s*8,px);
  };
  auto load=[&](int i) { loadw(i); loadx(i); };'''
    d.edit(old, new)
    old = '    for(int i=0;i<ST && i<IT;i++) load(i);'
    new = '''    for(int i=0;i<ST && i<IT;i++) loadw(i);
    pdlo32_wait();
    #pragma unroll
    for(int i=0;i<ST && i<IT;i++) loadx(i);'''
    d.edit(old, new)
    old = '  fn<<<dim3(8192/R,(m+T-1)/T),128*TW+((PIPE&2)?128:0),SMEM,(cudaStream_t)stream>>>(tw,tx,(const uint8_t*)w,(__nv_bfloat16*)out,m);'
    new = '''  cudaLaunchConfig_t lc = {};
  lc.gridDim=dim3(8192/R,(m+T-1)/T);
  lc.blockDim=dim3(128*TW+((PIPE&2)?128:0));
  lc.dynamicSmemBytes=SMEM; lc.stream=(cudaStream_t)stream;
  cudaLaunchAttribute attr[1];
  attr[0].id=cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed=1;
  lc.attrs=attr; lc.numAttrs=1;
  const cudaError_t launch_rc=cudaLaunchKernelEx(&lc,fn,tw,tx,(const uint8_t*)w,(__nv_bfloat16*)out,m);
  if(launch_rc!=cudaSuccess) return int(launch_rc);'''
    d.edit(old, new)
    anchor = 'int ol_launch(int id,int64_t x,int64_t w,int64_t out,int m,int64_t stream,int query) {'
    d.edit(anchor, '''int pdlo32_info(int query) {
  if(query!=1) return -90;
  int blocks=0;
  const cudaError_t rc=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,ol_projection<32,128,8,1,1,1,1>,128,196736);
  return rc==cudaSuccess ? blocks : -static_cast<int>(rc);
}
''' + anchor)
    d.edit('ol_projection', 'pdlo32_projection', count=d.text.count('ol_projection'))
    return d.finish()


def derive():
    original, cpp = parents()
    OUT.mkdir(exist_ok=True)
    out = {}
    for name, make in [('o16', o16_twin), ('o32', o32_twin)]:
        candidate, proof = make(original[name])
        proof.update(parent_sha256=hashlib.sha256(original[name].encode()).hexdigest(),
                     candidate_sha256=hashlib.sha256(candidate.encode()).hexdigest())
        (OUT / (name + '-parent.cu')).write_bytes(original[name].encode())
        (OUT / (name + '-candidate.cu')).write_bytes(candidate.encode())
        (OUT / (name + '.cpp')).write_bytes(cpp[name].encode())
        out[name] = (original[name], candidate, proof)
    (OUT / 'binding.json').write_bytes((json.dumps({'decode_baseline_sha256': BASE_SHA,
        'GPU_jobs': 0, 'sources': {k: v[2] for k, v in out.items()}}, indent=2) + '\n').encode())
    return out


if __name__ == '__main__':
    for name, (old, new, proof) in derive().items():
        print(name, len(old), len(new), proof['candidate_sha256'], 'inverse', proof['exact_inverse'])
