"""One reversible FP32 finalizer-read reuse screen, never a kernel launch."""
import ast
import hashlib
import json
from pathlib import Path

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'STATE.json').exists())
PARENT='331148716e8d1ba92b45cd29725be997d44b3a5235af74d75770bb9444100d50'
CUDA='1e60cf469ebff8d0f1e63a223f41f041374a1f30f90e4201a819ee178353270c'


def main():
    raw=(ROOT/'best/moe_verified.py').read_bytes()
    assert hashlib.sha256(raw).hexdigest()==PARENT
    tree=ast.parse(raw)
    original=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
                  and any(isinstance(t,ast.Name) and t.id=='CUDA_SRC' for t in n.targets))
    assert hashlib.sha256(original.encode()).hexdigest()==CUDA
    source=original
    edits=[]
    def edit(name,old,new,count=1):
        nonlocal source
        assert source.count(old)==count,(name,source.count(old))
        source=source.replace(old,new)
        edits.append(dict(name=name,old=old,new=new,count=count))
    edit('namespace','namespace okc {','namespace okc_f32cache {')
    edit('namespace comment','}  // namespace okc','}  // namespace okc_f32cache')
    edit('host references','okc::','okc_f32cache::',source.count('okc::'))
    start=source.index('#if OKC_WGS && OKC_WGSFINSS\n')
    stop=source.index('\n#elif',start)
    old=source[start:stop]
    # Same per-thread FP32 values survive the norm reduction; no new format.
    new=old.replace('      float ps[8];','      float ps[8];\n      float4 saved_a[8], saved_b[8];',1)
    load='          const float4 a = __ldcg(src), b = __ldcg(src + 1);'
    assert new.count(load)==2
    first='          const float4 a = saved_a[sl] = __ldcg(src), b = saved_b[sl] = __ldcg(src + 1);'
    new=new.replace(load,first,1)
    # The second pass does not need another address or global load.
    second='''          const float4* src = reinterpret_cast<const float4*>(p.ybuf + (size_t)qv[sl] * HD) + tid * 2;
          const float4 a = __ldcg(src), b = __ldcg(src + 1);'''
    assert new.count(second)==1
    new=new.replace(second,'          const float4 a = saved_a[sl], b = saved_b[sl];',1)
    edit('retain original FP32 row fragments through finalizer',old,new)
    # All-nonlocal diagnostic inputs must not read an uninitialized job row.
    edit('empty job guard',
         '__device__ __forceinline__ int wgs_stage_at(const Smem& S, int nj, int pb, uint32_t x) {\n  const int* pre',
         '__device__ __forceinline__ int wgs_stage_at(const Smem& S, int nj, int pb, uint32_t x) {\n  if (nj == 0) return 0;\n  const int* pre')
    edit('empty range guard',
         '__device__ __forceinline__ int wgs_split(int pb, int g0, int g1) {\n  const int ue',
         '__device__ __forceinline__ int wgs_split(int pb, int g0, int g1) {\n  if (g1 <= g0) return g0;\n  const int ue')
    inverse=source
    for e in reversed(edits):
        assert inverse.count(e['new'])==e['count'],e['name']
        inverse=inverse.replace(e['new'],e['old'])
    assert inverse==original
    prefix=source.split('#include <torch/extension.h>',1)[0]
    baseline=original.split('#include <torch/extension.h>',1)[0]
    out=ROOT/'.tools/moe-fp32-final-cache'
    out.mkdir(exist_ok=True)
    for name,text in [('full.cu',source),('device.cu',prefix),('baseline.cu',baseline)]:
        (out/name).write_bytes(text.encode())
    (out/'edits.json').write_text(json.dumps(edits,indent=2)+'\n',encoding='utf-8')
    report=dict(parent_sha256=PARENT,parent_CUDA_sha256=CUDA,
                full_CUDA_sha256=hashlib.sha256(source.encode()).hexdigest(),
                device_prefix_sha256=hashlib.sha256(prefix.encode()).hexdigest(),
                reversible_edits=len(edits),inverse_recovers_entire_CUDA=True,
                old_FP32_load_sites_per_unrolled_route=2,new_FP32_load_sites_per_unrolled_route=1,
                saved_FP32_values_per_consumer_thread=64,added_global_or_shared_scratch_bytes=0,
                no_norm_gain_movement_or_BF16_conversion=True,
                limitations=['No exposed finalizer latency measurement',
                             'Register pressure may spill or force compiler reloads',
                             'Source-level reads are not measured L2/HBM traffic'],
                GPU_jobs=0,kernel_execution=False)
    (out/'source-audit.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report))


if __name__=='__main__':main()
