"""One bounded repair: match scalar and tiny-Triton reductions separately."""
import hashlib
import json
import re
import subprocess
import sys
from preflight_decoder_cluster_norm import ROOT,BASE,freeze

OUT=ROOT/'results/decoder-cluster-norm-dual-order-preflight-20261029'


def main():
    original=(ROOT/'experiments/prepared/decoder_cluster_norm_ordered.cu').read_bytes()
    assert hashlib.sha256(original).hexdigest()=='49bf73e68a8713a431f11d03d6f2f20e361562d4a105140962f8b6a11f269634'
    source=original.decode();edits=[]
    def edit(a,b,count=1):
        nonlocal source
        assert source.count(a)==count,(a,source.count(a))
        source=source.replace(a,b);edits.append(dict(old=a,new=b,count=count))
    edit('namespace okc_cluster_norm_ordered','namespace okc_cluster_norm_dual_order',2)
    edit('template<int PHASE>','template<int PHASE,bool TRITON>')
    edit('  return rsqrtf(sum*(1.f/H)+eps);',
         '  if constexpr (TRITON) return rsqrtf(__fadd_rn(__fmul_rn(sum,1.f/H),eps));\n  else return rsqrtf(sum*(1.f/H)+eps);')
    anchor='//0attention/routerFP32,1attention/no router,2FFN slicedT<=8,3fullFFN.'
    helper='''// The retained Triton tiny-FFN uses separately rounded squares and a
// balanced adjacent-pair tree over32 strided elements/thread. Its original
// 256 logical lanes are distributed exactly as for the scalar modes.
template<int BEGIN,int COUNT>
__device__ __forceinline__ float tiny_tree(const float (&values)[PACKS][VEC]) {
  if constexpr (COUNT==1) {
    const float v=values[BEGIN/VEC][BEGIN%VEC];
    return __fmul_rn(v,v);
  } else {
    return __fadd_rn(tiny_tree<BEGIN,COUNT/2>(values),tiny_tree<BEGIN+COUNT/2,COUNT/2>(values));
  }
}

'''
    edit(anchor,helper+anchor)
    edit('const int col=vector_index*VEC+v;',
         'const int col=SLICE ? logical_thread+(pack*VEC+v)*LOGICAL_THREADS : vector_index*VEC+v;',3)
    edit('      ss=fmaf(b,b,ss);','      if constexpr (!SLICE) ss=fmaf(b,b,ss);')
    edit('  const float post=row_inverse<0>(ss,s,cluster,p.post_eps);',
         '  if constexpr (SLICE) ss=tiny_tree<0,PACKS*VEC>(values);\n  const float post=row_inverse<0,SLICE>(ss,s,cluster,p.post_eps);')
    edit('      pre_ss=fmaf(value,value,pre_ss);','      if constexpr (!SLICE) pre_ss=fmaf(value,value,pre_ss);')
    edit('  const float pre=row_inverse<1>(pre_ss,s,cluster,p.pre_eps);',
         '  if constexpr (SLICE) pre_ss=tiny_tree<0,PACKS*VEC>(values);\n  const float pre=row_inverse<1,SLICE>(pre_ss,s,cluster,p.pre_eps);')
    recovered=source
    for e in reversed(edits):
        assert recovered.count(e['new'])==e['count'];recovered=recovered.replace(e['new'],e['old'])
    assert recovered.encode()==original
    path=ROOT/'experiments/prepared/decoder_cluster_norm_dual_order.cu'
    freeze(path,source.encode());sha=hashlib.sha256(source.encode()).hexdigest()
    work=ROOT/'.tools/decoder-cluster-norm-dual-order-v1/main'
    if not work.with_suffix('.json').exists():
        cmd=[sys.executable,str(ROOT/'experiments/prepared/local_nvrtc_compile.py'),'--source',str(path),'--out',str(work)]
        for k in range(4):cmd+=['--name',f'&okc_cluster_norm_dual_order::boundary<{k}>']
        subprocess.run(cmd,cwd=ROOT,check=True)
    meta=json.loads(work.with_suffix('.json').read_bytes());assert meta['return_code']==0 and meta['original_source_sha256']==sha
    log=work.with_suffix('.log').read_text();registers=list(map(int,re.findall(r'Used (\d+) registers',log)))
    spills=re.findall(r'(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads',log)
    assert len(registers)==4 and max(registers)<=64,(registers,'native resource gate unchanged')
    assert spills and all(tuple(map(int,r))==(0,0,0) for r in spills)
    for ext in ('.cu','.json','.log'):freeze(OUT/'v1'/('compiler'+ext),work.with_suffix(ext).read_bytes())
    report=dict(parent_source_sha256=BASE,native_CUDA_sha256=sha,offline_register_counts=registers,
        zero_stack_and_spills=True,static_shared_bytes=16,CPU_only=True,GPU_jobs=0,
        source_parent_sha256=hashlib.sha256(original).hexdigest(),edits=edits,
        inverse_restores_parent=True,evaluator_Triton_version_unknown=True,
        tiny_mapping_evidence='Triton3.9.0, exact retained AST, explicit sm90,8warps,FP fusion disabled; artifacts in decoder-sfn-offline-reduction-20261029',
        all_four_semantic_modes_preserved=True,numerical_limits_unchanged=True)
    freeze(OUT/'v1/binding.json',(json.dumps(report,indent=2)+'\n').encode())
    print(json.dumps({k:v for k,v in report.items() if k!='edits'},indent=2))


if __name__=='__main__':main()
