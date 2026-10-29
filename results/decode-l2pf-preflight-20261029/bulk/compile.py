import sys, subprocess
sys.path.insert(0, 'results/decode-l2pf-preflight-20261029/bulk')
import triton, pfb_src
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
k = triton.compile(ASTSource(pfb_src._pfb, signature={'T': '*i64', 'W': 'constexpr', 'CH': 'constexpr'}, constexprs={'W': 1, 'CH': 65536}), target=GPUTarget('cuda', 90, 32), options={'num_warps': 1})
open('results/decode-l2pf-preflight-20261029/bulk/pfb.cubin','wb').write(k.asm['cubin'])
ptx = k.asm['ptx']
print('ptx bulk prefetch:', ptx.count('cp.async.bulk.prefetch.L2.global'))
import pathlib
tool = pathlib.Path(triton.__file__).parent / 'backends/nvidia/bin/cuobjdump'
s = subprocess.run([str(tool), '-sass', 'results/decode-l2pf-preflight-20261029/bulk/pfb.cubin'], capture_output=True, text=True).stdout
print([l.strip()[:90] for l in s.splitlines() if 'UBLKPF' in l or 'BLKPF' in l or 'UTMA' in l or 'PF' in l][:6])
