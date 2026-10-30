import re
import sys
from pathlib import Path

sys.path.insert(0, r"C:\Users\Achita\Documents\ChatGPT\Muse 2\experiments\prepared")
import local_nvrtc_compile as L

tpl = Path(r'C:\Users\Achita\AppData\Local\Temp\opencode\v18d_template.cu').read_text(encoding='utf-8')
dev = tpl.split('static int v18d_native_smem_optin()')[0]
for inc in ('#include <cuda_runtime.h>\n', '#include <algorithm>\n', '#include <stdint.h>\n', '#include <type_traits>\n'):
    dev = dev.replace(inc, '')
dev = ('#include <cuda/std/cstdint>\n#include <cuda/std/type_traits>\nusing namespace cuda::std;\n'
       'namespace std { using cuda::std::conditional; }\n' + dev)
insts = [(1, 1, 1, 2, 2, d, 256, 0, 2304, 8192, 1, 4, 8, 4) for d in (1, 2, 4, 8, 16)] + [(1, 1, 1, st, 2, 0, 256, 0, 2304, 8192, 1, 4, 8, 4) for st in (3, 4)]
names = ['&v18d_native_gemv<' + ', '.join(str(v) for v in i) + '>' for i in insts]
flags = ['--gpu-architecture=sm_90a', '--std=c++20', '--device-as-default-execution-space', '--ptxas-options=-v',
         '--include-path=' + str(L.TOOL / 'nvidia/cu13/include'),
         '--include-path=' + str(L.TOOL / 'nvidia/cu13/include/cccl')]
rc, log, cubin, dur = L.NVRTC().compile(dev.encode(), names, flags, False)
out = Path('results/p3-offline')
out.mkdir(parents=True, exist_ok=True)
(out / 'nvrtc.log').write_text(log, encoding='utf-8')
lines = [f'rc {rc} secs {dur:.1f}']
for b in re.split(r"(?=ptxas info\s*: Compiling entry function)", log):
    st = re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", b)
    rg = re.search(r"Used (\d+) registers", b)
    m = re.search(r"v18d_native_gemvILi" + r"ELi".join([r"(\d+)"] * 14) + "E", b)
    if st and rg:
        lines.append(f'  {m.groups() if m else "?"} regs {rg.group(1)} stack/spill {st.groups()}')
if rc:
    lines.append(log[:2000])
(out / 'summary.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
