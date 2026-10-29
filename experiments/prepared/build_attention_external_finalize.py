"""Fixed L8-flat external merge draft; preserve all numerical mainloop code."""
import ast
import hashlib
import json
from pathlib import Path

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'STATE.json').exists())
PARENT='9e922c3555484081ca45a99a90284e26eccab181fb677700d8227a4b3402412f'
CUDA='241b67fe0008bbae53435b53d1bb6d647ded0b5718118875c0ec368b718b41db'
OUT=ROOT/'results/attention-external-finalize-preflight-20261029/v1'

KERNEL=r'''
// A normal stream edge joins the completed atomic partials. Each CTA owns one
// token/KV-pair's sixteen heads; no grid barrier or cross-CTA spin is needed.
__global__ void __launch_bounds__(128)
attn_external_finalize8(const __grid_constant__ Params p) {
  const int token = blockIdx.x, pair = blockIdx.z;
  const int pairs = p.Hkv >> 1;
  const int request = token >> 3, query = token & 7;
  const int workspace_row = (request * pairs + pair) * 8 + query;
  const float vscale = __ldg(p.v_scale);
  // L=1 is only an indexing adaptation here: token indexes the original
  // flattened [B,8] qkv/output while workspace_row indexes [B,pairs,8].
  // All sixteen heads and their full64-element norm domains stay intact.
  external_fin_nc1<1,128>(p,token,pair,workspace_row,vscale,0);
}
'''


def sha(raw): return hashlib.sha256(raw).hexdigest()


def freeze(path,raw):
    path.parent.mkdir(parents=True,exist_ok=True)
    assert not path.exists() or path.read_bytes()==raw,'Frozen draft collision: '+str(path)
    path.write_bytes(raw)


def main():
    raw=(ROOT/'best/attention_checked.py').read_bytes();assert sha(raw)==PARENT
    tree=ast.parse(raw)
    assignment=next(n for n in tree.body if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='CUDA_SRC' for t in n.targets))
    original=ast.literal_eval(assignment.value);assert sha(original.encode())==CUDA
    source=original;edits=[]

    def replace(old,new,count=1,label=''):
        nonlocal source
        assert source.count(old)==count,(label,source.count(old),count)
        source=source.replace(old,new)
        edits.append(dict(label=label,old=old,new=new,count=count))

    # Preserve the full end-of-segment CTA barrier: the producer waits there
    # after draining its cp.async copies. Only completion/publication is moved.
    begin='  // (last-CTA finalizes): no per-thread fences; one acq_rel atomic by thread 0 after the CTA barrier'
    end="     // consumers' shared memory is reused by the next segment"
    replace(begin,'  if constexpr (!(FLAT && L == 8 && (MODE & 16))) {\n'+begin,label='skip_external_counter_and_in_kernel_finalization')
    replace(end,'  } // external merge preserves the following producer/consumer CTA join\n'+end,label='retain_full_segment_end_barrier')
    replace('if constexpr (FLAT && (L == 8)) {\n    if (p.coop && threadIdx.x == 0)',
            'if constexpr (FLAT && (L == 8) && !(MODE & 16)) {\n    if (p.coop && threadIdx.x == 0)',label='skip_unused_external_epoch_copy')
    replace('if (p.coop) {   // still part of the grid-wide finalize',
            'if (p.coop && !(MODE & 16)) {   // still part of the grid-wide finalize',label='empty_external_CTA_needs_no_grid_wait')
    replace('if (p.coop) flat_finalize<','if (p.coop && !(MODE & 16)) flat_finalize<',count=2,label='skip_external_flat_grid_join')
    replace('segment<L, FLAT, R>(p, S,','segment<L, FLAT, R, MODE>(p, S,',label='propagate_fixed_external_mode_to_segments')

    anchor='template <int L, int NT>\n__device__ __forceinline__ void fin_nc1('
    start=original.index(anchor);finish=original.index('\n}\n',start)+3
    finalizer=original[start:finish]
    assert finalizer.count('fin_nc1(')==1
    cloned=finalizer.replace('fin_nc1(','external_fin_nc1(',1)
    loaded='    }\n#pragma unroll\n    for (int u = 0; u < GP; ++u) {\n      const int rr = (p0 + u)'
    assert cloned.count(loaded)==1
    cloned=cloned.replace(loaded,'    }\n    __syncwarp(0xffffffffu); // all eight lanes read lsum before part0 clears it\n#pragma unroll\n    for (int u = 0; u < GP; ++u) {\n      const int rr = (p0 + u)',1)
    assert cloned.replace('external_fin_nc1(','fin_nc1(',1).replace(
        '    __syncwarp(0xffffffffu); // all eight lanes read lsum before part0 clears it\n','',1)==finalizer
    replace(finalizer,finalizer+'\n'+cloned+KERNEL,label='parallel_token_finalizer_reuses_exact_retained_norm_math')

    replace('  static int occ_c = 0, occ_f = 0, nsm = 0;',
            '  static int occ_c = 0, occ_f = 0, nsm = 0;\n  static int external_occ = 0;',label='native_external_resource_slot')
    init='    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ_f, attn_kernel_fp8pv<L, true>, Cfg<L, true>::NTX, sm_f);'
    extra=r'''
    if constexpr (L == 8) {
      auto external = attn_kernel_fp8pv<8,true,16>;
      cudaError_t err = cudaFuncSetAttribute(external,cudaFuncAttributeMaxDynamicSharedMemorySize,sm_f);
      TORCH_CHECK((err == cudaSuccess),"external attention shared-memory admission failed");
      err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&external_occ,external,Cfg<8,true>::NTX,sm_f);
      TORCH_CHECK((err == cudaSuccess),"external attention occupancy query failed");
    }
'''
    replace(init,init+extra,label='configure_exact_external_main_instance')
    dispatch='  if (cl || pdl) {\n    // cluster dims and / or programmatic stream serialization'
    extra_launch=r'''  if constexpr (L == 8) {
    // Match only retained full-context cooperative flat/nc1 launches. All
    // shapes outside this fixed family and all PDL/debug calls remain old.
    if (p.flat && q.coop && p.window == 0 && p.B > 1 && p.nc == 1 &&
        !pdl && !p.dbg && external_occ >= 2) {
      attn_kernel_fp8pv<8,true,16><<<grid,Cfg<8,true>::NTX,sm_f,stream>>>(q);
      TORCH_CHECK((cudaGetLastError() == cudaSuccess),"external attention main launch failed");
      attn_external_finalize8<<<dim3(p.B*8,1,p.Hkv/2),128,0,stream>>>(q);
      TORCH_CHECK((cudaGetLastError() == cudaSuccess),"external attention finalizer launch failed");
      return;
    }
  }
'''
    replace(dispatch,extra_launch+dispatch,label='normal_stream_dependency_and_parallel_output_pass')
    recovered=source
    for edit in reversed(edits):
        assert recovered.count(edit['new'])==edit['count'],edit['label']
        recovered=recovered.replace(edit['new'],edit['old'])
    assert recovered==original
    # The FP8 PV, BF16 Q/K/RoPE/output and all key/segment ownership arithmetic
    # stays identical. Replace the one literal without importing GPU code.
    lines=raw.splitlines(keepends=True);node=assignment.value
    a=sum(map(len,lines[:node.lineno-1]))+node.col_offset
    b=sum(map(len,lines[:node.end_lineno-1]))+node.end_col_offset
    candidate=raw[:a]+repr(source).encode()+raw[b:]
    updated=ast.parse(candidate)
    n=next(n for n in updated.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CUDA_SRC' for t in n.targets))
    assert ast.literal_eval(n.value)==source
    n.value=assignment.value
    assert ast.dump(updated)==ast.dump(tree)
    report=dict(parent_source_sha256=PARENT,parent_CUDA_sha256=CUDA,source_sha256=sha(candidate),
        CUDA_sha256=sha(source.encode()),bytes=len(candidate),exact_inverse_restores_retained_CUDA=True,
        Python_AST_unchanged=True,edits=edits,scope='L8 full,flat,nc1,cooperative baseline selection,B>1,no PDL/debug,external occupancy>=2',
        untouched='L1, L8 sliding/single-request/other copies/PDL/debug/non-native fallbacks',
        extra_scratch_bytes=0,extra_timed_launches=1,extra_expanded_weights=False,
        full_end_of_segment_barriers_retained=True,all_numerical_mainloop_source_unchanged=True,
        CPU_only=True,GPU_jobs=0,admitted_for_GPU=False,
        remaining='ownership/replay/numerical traffic model, compiler/resource/control-body proof, fresh evaluator identity, native resource and execution guards')
    freeze(OUT/'submission.py',candidate);freeze(OUT/'kernel.cu',source.encode());freeze(OUT/'baseline.cu',original.encode())
    freeze(OUT/'binding.json',(json.dumps(report,indent=2)+'\n').encode())
    print(json.dumps({k:v for k,v in report.items() if k!='edits'},indent=2))


if __name__=='__main__':main()
