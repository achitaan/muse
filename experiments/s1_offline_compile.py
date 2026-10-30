"""Offline NVRTC sm_90a resource check for the S1 v18s_gemv instances (never launches a kernel)."""
import ast
import base64
import re
import sys
import zlib
from pathlib import Path

sys.path.insert(0, r"C:\Users\Achita\Documents\ChatGPT\Muse 2\experiments\prepared")
import local_nvrtc_compile as L

src = Path('best/verified.py').read_text(encoding='utf-8')
t = ast.parse(src)
tpl = [zlib.decompress(base64.b64decode(ast.literal_eval(n.value.args[0]))).decode()
       for n in t.body if isinstance(n, ast.Assign) and getattr(n.targets[0], 'id', '') == 'V18S_CUDA_TEMPLATE'][0]
dev = tpl.split('static int v18s_smem_optin()')[0]
for inc in ('#include <cuda_runtime.h>\n', '#include <algorithm>\n', '#include <stdint.h>\n', '#include <type_traits>\n'):
    dev = dev.replace(inc, '')
dev = ('#include <cuda/std/cstdint>\n#include <cuda/std/type_traits>\nusing namespace cuda::std;\n'
       'namespace std { using cuda::std::conditional; }\n' + dev)
ns = {}
exec(Path('experiments/dense_s1_addon.py').read_text(encoding='utf-8').split('def _s1_graph')[0], ns)
insts = [ns['_s1_inst'](g)[0] for g in ns['_S1_GEOMS']]
names = ['&v18s_gemv<' + ', '.join(str(v) for v in i) + '>' for i in insts]
flags = ['--gpu-architecture=sm_90a', '--std=c++20', '--device-as-default-execution-space', '--ptxas-options=-v',
         '--include-path=' + str(L.TOOL / 'nvidia/cu13/include'),
         '--include-path=' + str(L.TOOL / 'nvidia/cu13/include/cccl')]
rc, log, cubin, dur = L.NVRTC().compile(dev.encode(), names, flags, False)
out = Path('results/s1-offline')
out.mkdir(parents=True, exist_ok=True)
(out / 'nvrtc-s1.log').write_text(log, encoding='utf-8')
lines = [f'insts {insts} rc {rc} secs {dur:.1f}']
for b in re.split(r"(?=ptxas info\s*: Compiling entry function)", log):
    st = re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", b)
    rg = re.search(r"Used (\d+) registers", b)
    sm = re.search(r"(\d+) bytes smem", b)
    m = re.search(r"ELi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)EEv", b)
    if st and rg:
        lines.append(f'  inst {m.groups() if m else "?"} regs {rg.group(1)} stack/spill_st/spill_ld {st.groups()} smem {sm.group(1) if sm else 0}')
if rc != 0:
    lines.append(log[:2000])
(out / 's1-resources.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
print('\n'.join(lines))
