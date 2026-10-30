"""Pure, invertible derivations for the bounded T1/T8 PDL integration."""
import hashlib
import re
from pdl_source import edit_once, twin


def pdli_trigger(source):
    """Every producer CTA announces readiness at entry; no arithmetic changes."""
    start = source.index('{', source.index('__global__')) + 1
    add = '\n  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");\n'
    return source[:start] + add + source[start:], {'offset': start, 'added': add}


def pdli_unpool(source, inst):
    assert inst in ((1,1,1,2,2,0,256,1,1024,4096,1,8,2,4),
                    (1,1,1,2,2,0,512,8,1024,4096,1,16,2,2))
    edits = []
    original = source
    source = edit_once(source, 'ld.global.nc.v4.u32', 'ld.global.cg.v4.u32', edits)
    xloads = '''      if constexpr (!NOX && !RO) {
#pragma unroll
        for (int n = 0; n < MT; ++n) {
          const __nv_bfloat16* xp = x0 + (size_t)min(n * 8 + (lane >> 2), m - 1) * K + s * 64;
          xv[slot][n][0] = smtrace_ldx(xp);
          xv[slot][n][1] = smtrace_ldx(xp + 8);
        }
      }'''
    source = edit_once(source, xloads, '      // PDL: initial activations are loaded after the dependency wait.', edits)
    helper = '''__device__ __noinline__ void smtrace_pdl_wait(uint32_t checksum) {
  asm volatile("{ .reg .b32 used; mov.b32 used, %0; griddepcontrol.wait; }" :: "r"(checksum) : "memory");
}
'''
    anchor = '// Never taken, but ptxas cannot prove it:'
    source = edit_once(source, anchor, helper + '\n' + anchor, edits)
    words = ['(uint32_t)w[%d][0][%d].%s' % (s,l,c) for s in range(2) for l in range(3) for c in 'xyzw']
    old = '#pragma unroll\n    for (int s = 0; s < ST; ++s) load(s, s);'
    new = ('static_assert(FMT == 1 && MT == 1 && RT == 1 && ST == 2 && CKU == ST && DIAG == 0, "fixed full-split PDL preamble");\n'
           '#pragma unroll\n    for (int s = 0; s < ST; ++s) load(s, s);\n'
           '    smtrace_pdl_wait(my_e ^ ' + ' ^ '.join(words) + ');\n'
           '#pragma unroll\n    for (int s = 0; s < ST; ++s) {\n' + xloads.replace('xv[slot]', 'xv[s]') + '\n    }')
    source = edit_once(source, old, new, edits)
    source = edit_once(source, 'cudaLaunchAttribute at[1];', 'cudaLaunchAttribute at[2];', edits)
    attr = '''  const int pdl_attr = lc.numAttrs;
  at[pdl_attr].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[pdl_attr].val.programmaticStreamSerializationAllowed = 1;
  lc.attrs = at; lc.numAttrs = pdl_attr + 1;
'''
    source = edit_once(source, '  if (query == 1) {', attr + '  if (query == 1) {', edits)
    inverse = source
    for a, b in reversed(edits):
        assert inverse.count(b) == 1
        inverse = inverse.replace(b, a)
    assert inverse == original
    return source, {'exact_inverse':True, 'edits':[{'old':a,'new':b} for a,b in edits]}


def pdli_export(source, prefix, inst, *, fused=False):
    # Occupancy queries use the original cluster-only config. PDL affects ordering,
    # not resource usage; attach it only on a real launch.
    attr = '''  const int pdl_attr = lc.numAttrs;
  at[pdl_attr].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[pdl_attr].val.programmaticStreamSerializationAllowed = 1;
  lc.attrs = at; lc.numAttrs = pdl_attr + 1;
'''
    if attr in source:
        assert source.count(attr)==1
        source=source.replace(attr,'  if (query == 0) {\n'+attr+'  }\n')
    args = ','.join(map(str,inst))
    if not fused:
        source += ('\nint pdli_project(int64_t wp,int64_t x,int64_t out,int64_t esc,int m,int N,int K,int WR,int WK,int KU,int S,'
                   'int64_t t01,int64_t u01,int64_t stream){return ' + prefix + '_launch_t<' + args +
                   '>(wp,x,out,esc,m,N,K,WR,WK,KU,S,t01,u01,stream,0);}\n')
    source += ('\nint pdli_info(int m,int query){if(query!=1&&query!=2)return -90;return ' + prefix + '_launch_t<' + args +
               '>(0,0,0,0,m,' + str(inst[8]) + ',' + str(inst[9]) + ',' + ','.join(map(str,inst[10:14])) +
               ',0,0,0,query' + (',R2Args{}' if fused else '') + ');}\n')
    return source


def pdli_make_sources():
    items = [
        ('o1','smallm2',tt_v10_sm2_SMALLM2_CUDA_TEMPLATE,V12_SM2_O1['o',1][0],'o'),
        ('o8','v18s',V18S_CUDA_TEMPLATE,V18_SMALL['o',8][0],'o'),
        ('r1','v18s',_r2_source(V18S_CUDA_TEMPLATE,'v18s',V18_SMALL['pooler',1][0]),
         V18_SMALL['pooler',1][0][:-1]+(V18_SMALL['pooler',1][0][-1]|64,),'r'),
        ('r8','smallm2',_r2_source(tt_v10_sm2_SMALLM2_CUDA_TEMPLATE,'smallm2',tt_v10_sm2_SMALLM2_ROUTES['pooler',8][0]),
         tt_v10_sm2_SMALLM2_ROUTES['pooler',8][0][:-1]+(tt_v10_sm2_SMALLM2_ROUTES['pooler',8][0][-1]|64,),'r'),
        ('u1','smtrace',tt_v12_smtrace1_SMTRACE_CUDA_TEMPLATE,tt_v12_smtrace1_SMTRACE_ROUTES['unpooler',1][0],'u'),
        ('u8','smtrace',tt_v13_smtrace8_SMTRACE_CUDA_TEMPLATE,tt_v13_smtrace8_SMTRACE_ROUTES['unpooler',8][0],'u'),
        ('q1','v18d_native',v18d_SM_CUDA_TEMPLATE,(1,1,1,2,2,0,256,0,2304,8192,1,4,8,4),'q'),
        ('q8','qs_literal_owner5_decode5',BUNDLE_QKV_CUDA,(1,1,1,2,5,0,256,8,2304,8192,1,4,8,4),'q'),
    ]
    sources, specs, records = {}, {}, {}
    for name, prefix, source, inst, kind in items:
        source = source.replace('@@CASES@@','')
        parent_hash = hashlib.sha256(source.encode()).hexdigest()
        if kind in ('o','r'):
            source, proof = twin(source,prefix,inst,fused=kind=='r',form='call')
        elif kind == 'u':
            source, proof = pdli_unpool(source,inst)
        else:
            proof = {'arithmetic_unchanged':True}
        source, trig = pdli_trigger(source)
        source = pdli_export(source,prefix,inst,fused=kind=='r')
        sources[name] = source
        specs[name] = {'prefix':prefix,'instance':inst,'kind':kind}
        records[name] = {'parent_CUDA_sha256':parent_hash,'CUDA_sha256':hashlib.sha256(source.encode()).hexdigest(),
                         'derivation':proof,'trigger':trig,'occupancy_uses_cluster_only_config':True}
    return sources, specs, records


def pdli_pack(sources,specs):
    """One extension shares CUDA headers and the pybind translation unit across twins."""
    headers = ('#include <cuda_runtime.h>\n#include <cuda_bf16.h>\n#include <stdint.h>\n'
               '#include <algorithm>\n#include <type_traits>\n#include <cooperative_groups.h>\n')
    cuda = headers
    cpp = '#include <pybind11/pybind11.h>\n#include <stdint.h>\n'
    functions = []
    proj_types = ['int64_t']*4 + ['int']*7 + ['int64_t']*3
    r_types = ['int64_t']*4 + ['int']*7 + ['int64_t']*10 + ['int']
    for name,source in sources.items():
        includes = set(re.findall(r'^\s*#include\s+<([^>]+)>',source,re.M))
        assert includes <= {'cuda_runtime.h','cuda_bf16.h','stdint.h','algorithm','type_traits','cooperative_groups.h'},includes
        body = re.sub(r'^\s*#include\s+<[^>]+>[^\n]*\n','',source,flags=re.M)
        cuda += '\nnamespace pdli_'+name+' {\n'+body+'\n}\n#undef SMD\n'
        if '#define DC2_DECODE' in body:
            cuda += '#undef DC2_DECODE\n'
        for suffix,types,target in (
            ('project',r_types if specs[name]['kind']=='r' else proj_types,
             'r2_launch' if specs[name]['kind']=='r' else 'pdli_project'),
            ('info',['int','int'],'pdli_info'),
        ):
            fn = 'pdli_'+name+'_'+suffix
            args = ','.join(t+' a'+str(i) for i,t in enumerate(types))
            call = ','.join('a'+str(i) for i in range(len(types)))
            cuda += 'int '+fn+'('+args+'){return pdli_'+name+'::'+target+'('+call+');}\n'
            cpp += 'int '+fn+'('+','.join(types)+');\n'
            functions.append(fn)
    return cuda,cpp,functions
