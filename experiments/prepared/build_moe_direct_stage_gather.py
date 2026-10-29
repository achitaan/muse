"""One reversible MAIN/WGS direct-gather CUDA draft; never loads a GPU."""
import ast
import argparse
import hashlib
import json
from pathlib import Path

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'STATE.json').exists())
BASE='331148716e8d1ba92b45cd29725be997d44b3a5235af74d75770bb9444100d50'
NATIVE='1e60cf469ebff8d0f1e63a223f41f041374a1f30f90e4201a819ee178353270c'
OUT=ROOT/'results/moe-direct-stage-gather-preflight-20261029/v1'


def freeze(path,raw):
    path.parent.mkdir(parents=True,exist_ok=True)
    assert not path.exists() or path.read_bytes()==raw,'Frozen evidence collision: '+str(path)
    path.write_bytes(raw)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stripmine-copy',action='store_true',help='One source-level repair: lookup a row once for its eight copies')
    parser.add_argument('--publisher-completion',action='store_true',help='Complete/fence copies at the producer; restore the exact consumer loop')
    args=parser.parse_args()
    assert not args.publisher_completion or args.stripmine_copy
    out=OUT if not args.stripmine_copy else OUT.parent/'v2-stripmined-copy'
    if args.publisher_completion:out=OUT.parent/'v3-publisher-completion'
    raw=(ROOT/'best/moe_verified.py').read_bytes()
    assert hashlib.sha256(raw).hexdigest()==BASE
    tree=ast.parse(raw)
    original=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='CUDA_SRC' for t in n.targets))
    assert hashlib.sha256(original.encode()).hexdigest()==NATIVE
    source=original;edits=[]
    def edit(name,old,new,count=1):
        nonlocal source
        assert source.count(old)==count,(name,source.count(old),count)
        source=source.replace(old,new)
        edits.append(dict(name=name,old=old,new=new,count=count))
    edit('namespace','namespace okc {','namespace okc_direct {')
    edit('namespace comment','}  // namespace okc','}  // namespace okc_direct')
    edit('host references','okc::','okc_direct::',source.count('okc::'))
    a=source.index('__device__ void wgs_xproducer(')
    b=source.index('// phase A\'s x gather',a)
    old=source[a:b];new=old
    guard='  if ((threadIdx.x & 31) != 0) return;'
    assert new.count(guard)==1
    new=new.replace(guard,'  const int lane = threadIdx.x & 31;',1)
    wait='        if (!pb && !(OKC_WGSX & 8)) { while (ld_acquire(p.xready + j) < npad) __nanosleep(32); }  // every row gathered'
    assert new.count(wait)==1
    new=new.replace(wait,'        // Phase A reads fresh original activations directly; no xA publication wait.',1)
    copy_start=new.index('      const uint32_t bytes = (uint32_t)npad * 512u;')
    copy_end=new.index('      if (++xs == WGS_NX)',copy_start)
    bulk=new[copy_start:copy_end]
    direct=r'''      if (!pb) {
        const int count = S.jC[j];
        const int entry_start = S.jS[j];
        // uint4 destination order: [row/8][k8][row%8]. Every warp
        // iteration writes512 consecutive shared bytes; invalid rows are
        // zero-filled without reading an entry or constructing a bad pointer.
        for (int v = lane; v < npad * 32; v += 32) {
          const int row = (v >> 8) * 8 + (v & 7);
          const int k8 = (v >> 3) & 31;
          const bool valid = row < count;
          int token = 0;
          if (valid) token = S.ent[entry_start + row] >> 3;
          const __nv_bfloat16* input = p.x + (size_t)token * HD + st * 256 + k8 * 8;
          cp_async16(xring + xs * WGS_XS + v * 16, input, valid);
        }
        // Default form increments pending count before the deferred arrival.
        // All32 registrations precede the explicit release of initial count1.
        asm volatile("cp.async.mbarrier.arrive.shared.b64 [%0];\n" ::"r"(smem_u32(bar)) : "memory");
        __syncwarp(0xffffffffu);
        if (lane == 0) mbar_arrive1(bar);
      } else if (lane == 0) {
'''+bulk+'''      }
      __syncwarp(0xffffffffu);
'''
    if args.stripmine_copy:
        start=direct.index('        for (int v = lane;')
        end=direct.index('        // Default form',start)
        direct=direct[:start]+r'''        for (int r0 = 0; r0 < npad; r0 += 8) {
          const int row = r0 + (lane & 7);
          const bool valid = row < count;
          int token = 0;
          if (valid) token = S.ent[entry_start + row] >> 3;
          const __nv_bfloat16* input = p.x + (size_t)token * HD + st * 256 + (lane >> 3) * 8;
          unsigned char* output = xring + xs * WGS_XS + r0 * 512 + lane * 16;
#pragma unroll
          for (int k = 0; k < 8; ++k) cp_async16(output + k * 512, input + k * 32, valid);
        }
'''+direct[end:]
    if args.publisher_completion:
        start=direct.index('        // Default form')
        stop=direct.index('        __syncwarp(0xffffffffu);',start)
        direct=direct[:start]+r'''        // Complete this producer's generic-proxy writes, then publish their
        // async-proxy visibility before lane0 releases the shared stage.
        cp_commit();
        cp_wait<0>();
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
'''+direct[stop:]
    new=new[:copy_start]+direct+new[copy_end:]
    edit('all32 activation lanes; direct phaseA; original phaseB bulk',old,new)
    edit('remove initial global activation gather','  wgs_gather(p, S);',
         '  // Direct phase-A stage producer replaces global xA gathering.\n  // The following full producer/consumer WS_JOIN is retained.')
    # Non-bulk cp.async accesses the generic proxy. Every consumer orders the
    # acquired completed shared writes before its asynchronous WGMMA reads.
    anchor='    wgs_wait(S.bars + BX_F + xs, (uint32_t)(seq / WGS_NX) & 1u, 4000000 + seq);'
    if not args.publisher_completion:
        edit('consumer proxy ordering after completed direct copies',anchor,
             anchor+'\n    if constexpr (!PB) asm volatile("fence.proxy.async.shared::cta;\\n" ::: "memory");')
    # Preserve the same bounded all-nonlocal repairs used in previous admitted
    # MAIN candidates; no missing job/range may read uninitialized metadata.
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
    report=dict(parent_source_sha256=BASE,parent_CUDA_sha256=NATIVE,
        CUDA_sha256=hashlib.sha256(source.encode()).hexdigest(),bytes=len(source.encode()),
        exact_inverse_restores_entire_retained_CUDA=True,edits=edits,
        matrix_math_weights_epilogues_and_global_finalizer_unchanged=True,
        initial_global_gather_removed=True,phaseB_bulk_copy_unchanged=True,
        full_producer_consumer_join_retained=True,shared_layout_and_ring_capacity_unchanged=True,
        extra_scratch_bytes=0,old_scratch_allocation_retained=True,
        GPU_jobs=0,CPU_only=True,Python_submission_written=False,
        remaining='Compiler/resource and lifetime/replay proof, whole-load opportunity screen, and native numerical/performance validation')
    if args.stripmine_copy:
        report['repair']='v1 static spill burden rose to48/316/668 from16/56/124. Hoist repeated row lookup/address calculation across eight copies; no geometry/register/cache changes.'
    if args.publisher_completion:
        report['repair']='v2 still48/316/668. Complete generic-proxy copies and fence them in the producer; restore the exact consumer loop. This adds producer waiting and is not a performance claim.'
    freeze(out/'kernel.cu',source.encode());freeze(out/'baseline.cu',original.encode())
    freeze(out/'binding.json',(json.dumps(report,indent=2)+'\n').encode())
    print(json.dumps({k:v for k,v in report.items() if k!='edits'},indent=2))


if __name__=='__main__':main()
