"""Bounded L3 feasibility source, not a submission until resource targets pass."""
from pathlib import Path
import ast,hashlib
ROOT=Path(__file__).resolve().parents[1]
def load_cuda(path):
    t=path.read_bytes()
    return ast.literal_eval(next(n.value for n in ast.parse(t).body if isinstance(n,ast.Assign)
        and any(isinstance(v,ast.Name) and v.id=='CUDA_SRC' for v in n.targets)))
BASE=load_cuda(ROOT/'experiments/boards/moe/moe_b13.py')
L2=load_cuda(ROOT/'experiments/boards/moe/moe_l2.py')

def l3_source():
    src=L2
    assert '#define OKC_L2_CAP 80' in src and '#define OKC_L2_GS 1' in src
    a=src.index('    // codes and scales one 4-step group at a time',src.index('void wgs_seg('))
    b=src.index('    prev = xs;',a)
    old=src[a:b]
    inner=old[old.index('#pragma unroll\n      for (int h ='):]
    assert inner.endswith('    }\n');inner=inner[:-len('    }\n')]
    wide='''    if constexpr (NTT > 8) {
      // L3 NTT9/10: GS1 and only the CURRENT lg packed code/scale group.
      // No cwn/scn variables or next-lg lookahead exist in this instantiation.
#pragma unroll
      for (int lg = 0; lg < 4; ++lg) {
        const uint4 cwg = *reinterpret_cast<const uint4*>(pc + ((lg * 4 + wp) * 32 + lane) * 16);
        const uint2 scg = *reinterpret_cast<const uint2*>(pc + 8192 + ((lg * 4 + wp) * 8 + g) * 8);
        if (lg == 3) {
          const uint32_t loaded = cwg.x ^ cwg.y ^ cwg.z ^ cwg.w ^ scg.x ^ scg.y;
          // Real dependence on all last-group loads; p.zero is host-initialized0.
          const uint32_t mask = 0xffffffffu ^ (loaded & p.zero);
          asm volatile("bar.warp.sync %0;" :: "r"(mask) : "memory");
          if (lane == 0) mbar_arrive1(S.bars + BW_E + ws);
        }
'''+inner+'''      }
    } else {
'''+old+'''    }
'''
    src=src[:a]+wide+src[b:]
    assert 'setmaxnreg' not in src
    return src

if __name__=='__main__':
    src=l3_source();(ROOT/'.l3_offline/runtime.cu').write_text(src)
    print('L3 device+host source SHA',hashlib.sha256(src.encode()).hexdigest())
