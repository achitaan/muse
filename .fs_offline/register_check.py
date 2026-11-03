"""Bounded offline check of register allocation (same arithmetic/storage).

maxnreg224 is a NEGATIVE CONTROL, not a launchable candidate: CUDA13
cuda_occupancy.h:1540 rounds9 warps to12 for the hardware register check,
so224regs assumes86,016 registers, beyond65,536. Keep launch_bounds288.
"""
import os,sys,json
from pathlib import Path
os.environ['CUDA_CACHE_DISABLE']='1'
sys.dont_write_bytecode=True
from build_source import ROOT,source
sys.path.insert(0,str(ROOT.parents[2]/'experiments/prepared'))
import local_nvrtc_compile as nv
flags=['--gpu-architecture=sm_90a','--std=c++17','--device-as-default-execution-space',
       '--ptxas-options=-v','--include-path='+str(nv.TOOL/'nvidia/cu13/include'),
       '--include-path='+str(nv.TOOL/'nvidia/cu13/include/cccl')]
cu=source().split('#include <torch/extension.h>')[0]
variants={
    'maxnreg224':cu.replace('__launch_bounds__(NTL,1)','__maxnreg__(224)'),
    'bounds_only':cu.replace('__launch_bounds__(NTL,1)','__launch_bounds__(NTL)'),
    'half_unroll':cu.replace('      for(int half=0;half<2;++half,++seq)', '      #pragma unroll\n      for(int half=0;half<2;++half,++seq)'),
}
for name,code in variants.items():
    rc,log,_,secs=nv.NVRTC().compile(code.encode(),['&fs::kernel<false>','&fs::kernel<true>'],flags,True)
    (ROOT/'.fs_offline'/('reg_'+name+'.log')).write_text(log)
    print(name,rc,secs,log,flush=True)
