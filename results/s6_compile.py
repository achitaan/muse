import re
import sys
from pathlib import Path

sys.path.insert(0, r"C:\Users\Achita\Documents\ChatGPT\Muse 2\experiments\prepared")
import local_nvrtc_compile as L

tpl = Path(r'C:\Users\Achita\AppData\Local\Temp\opencode\v18s_template.cu').read_text(encoding='utf-8')
dev = tpl.split('static int v18s_smem_optin()')[0]
for inc in ('#include <cuda_runtime.h>\n', '#include <algorithm>\n', '#include <stdint.h>\n', '#include <type_traits>\n'):
    dev = dev.replace(inc, '')
dev = ('#include <cuda/std/cstdint>\n#include <cuda/std/type_traits>\nusing namespace cuda::std;\n'
       'namespace std { using cuda::std::conditional; }\n' + dev)
insts = []
for st in (2, 1):
    insts.append((1, 4, 2, st, 0, 0, 256, 0, 4096, 8192, 1, 8, 16, 1, 6))
    insts.append((1, 4, 2, st, 2, 0, 128, 0, 2304, 8192, 1, 4, 8, 4, 6))
names = ['&v18s_gemv<' + ', '.join(str(v) for v in i) + '>' for i in insts]
flags = ['--gpu-architecture=sm_90a', '--std=c++20', '--device-as-default-execution-space', '--ptxas-options=-v',
         '--include-path=' + str(L.TOOL / 'nvidia/cu13/include'),
         '--include-path=' + str(L.TOOL / 'nvidia/cu13/include/cccl')]
rc, log, cubin, dur = L.NVRTC().compile(dev.encode(), names, flags, False)
out = Path('results/s6-offline')
out.mkdir(parents=True, exist_ok=True)
(out / 'nvrtc.log').write_text(log, encoding='utf-8')
lines = [f'rc {rc} secs {dur:.1f}']
for b in re.split(r"(?=ptxas info\s*: Compiling entry function)", log):
    st = re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", b)
    rg = re.search(r"Used (\d+) registers", b)
    m = re.search(r"v18s_gemvILi" + r"ELi".join([r"(\d+)"] * 15) + "E", b)
    if st and rg:
        lines.append(f'  {m.groups() if m else "?"} regs {rg.group(1)} stack/spill {st.groups()}')
if rc:
    lines.append(log[:1500])
(out / 'summary.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
