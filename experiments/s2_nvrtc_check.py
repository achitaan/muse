"""Offline NVRTC sm_90a resource check for S2 (m32w_gemm at BN 64/128, M=128 unpooler). Never launches."""
import ast
import base64
import hashlib
import re
import sys
import zlib
from pathlib import Path

sys.path.insert(0, r"C:\Users\Achita\Documents\ChatGPT\Muse 2\experiments\prepared")
import local_nvrtc_compile as L

src = Path('best/verified.py').read_text(encoding='utf-8')
t = ast.parse(src)
ns = {'hashlib': hashlib, '_zsrc_b64': base64, '_zsrc_zlib': zlib}
want = {'dd_decodediet_tt_mw_mma', 'dd_decodediet_tt_mw_CUDA_PREFIX', 'dd_decodediet_tt_mw_CUDA_CORE', '_zsrc'}
for n in t.body:
    name = n.name if isinstance(n, ast.FunctionDef) else getattr(n.targets[0], 'id', '') if isinstance(n, ast.Assign) else ''
    if name in want:
        exec(compile(ast.Module([n], []), 'v', 'exec'), ns)
s2 = {}
exec(Path('experiments/dense_s2_addon.py').read_text(encoding='utf-8').split('# --- runtime ---')[0], s2)
prefix = ns['dd_decodediet_tt_mw_CUDA_PREFIX']
for inc in ('#include <cuda.h>\n', '#include <cuda_runtime.h>\n', '#include <stdint.h>\n'):
    prefix = prefix.replace(inc, '')
prefix = ('#include <cuda/std/cstdint>\nusing namespace cuda::std;\n'
          'struct alignas(64) CUtensorMap { unsigned long long opaque[16]; };\n' + prefix)
dev = prefix + '\n'.join(ns['dd_decodediet_tt_mw_mma'](n) for n in (16, 32, 64, 128)) + s2['_s2_core'](ns['dd_decodediet_tt_mw_CUDA_CORE'])
insts = s2['_S2_INSTS']
names = [f'&m32w_gemm<{bn},{r},{wk},{pf},{q},{str(c).lower()},{cap}>' for (bn, r, wk, pf, q, c, cap) in insts]
flags = ['--gpu-architecture=sm_90a', '--std=c++20', '--device-as-default-execution-space', '--ptxas-options=-v',
         '--include-path=' + str(L.TOOL / 'nvidia/cu13/include'),
         '--include-path=' + str(L.TOOL / 'nvidia/cu13/include/cccl')]
rc, log, cubin, dur = L.NVRTC().compile(dev.encode(), names, flags, False)
out = Path('results/s2-offline')
out.mkdir(parents=True, exist_ok=True)
(out / 'nvrtc-s2.log').write_text(log, encoding='utf-8')
lines = [f'insts {insts} rc {rc} secs {dur:.1f}']
for b in re.split(r"(?=ptxas info\s*: Compiling entry function)", log):
    st = re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", b)
    rg = re.search(r"Used (\d+) registers", b)
    m = re.search(r"m32w_gemmILi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)ELb(\d)ELi(\d+)E", b)
    if st and rg:
        lines.append(f'  inst {m.groups() if m else "?"} regs {rg.group(1)} stack/spill_st/spill_ld {st.groups()}')
if rc != 0:
    lines.append(log[:3000])
(out / 's2-resources.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
print('\n'.join(lines))
