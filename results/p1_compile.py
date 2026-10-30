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
ns = {}
exec(Path('experiments/dense_p1.py').read_text(encoding='utf-8').split('# --- runtime ---')[0], ns)
insts = []
for (g, red, nt) in ns['_P1_GEOMS']:
    for mt, ins in ns['_p1_insts'](g, red, nt).items():
        insts.append(ins + (0, 2304, 8192) + g)
names = ['&v18d_native_gemv<' + ', '.join(str(v) for v in i) + '>' for i in insts]
flags = ['--gpu-architecture=sm_90a', '--std=c++20', '--device-as-default-execution-space', '--ptxas-options=-v',
         '--include-path=' + str(L.TOOL / 'nvidia/cu13/include'),
         '--include-path=' + str(L.TOOL / 'nvidia/cu13/include/cccl')]
rc, log, cubin, dur = L.NVRTC().compile(dev.encode(), names, flags, False)
out = Path('results/p1-offline')
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
