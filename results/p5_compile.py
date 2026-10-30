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
base = ['--gpu-architecture=sm_90a', '--std=c++20', '--device-as-default-execution-space', '--ptxas-options=-v',
        '--include-path=' + str(L.TOOL / 'nvidia/cu13/include'),
        '--include-path=' + str(L.TOOL / 'nvidia/cu13/include/cccl')]
# (label, inst14, maxrreg or None)
CASES = [
    ('incumbent ST2 NT256', (1, 1, 1, 2, 2, 0, 256, 0, 2304, 8192, 1, 4, 8, 4), None),
    ('A (1,4,8,4) ST2 cap96', (1, 1, 1, 2, 2, 0, 128, 0, 2304, 8192, 1, 4, 8, 4), 96),
    ('B (1,4,8,4) ST4 cap102', (1, 1, 1, 4, 2, 0, 128, 0, 2304, 8192, 1, 4, 8, 4), 102),
    ('C (1,4,4,8) ST2 cap56', (1, 1, 1, 2, 2, 0, 128, 0, 2304, 8192, 1, 4, 4, 8), 56),
    ('D (1,4,4,8) ST4 cap56', (1, 1, 1, 4, 2, 0, 128, 0, 2304, 8192, 1, 4, 4, 8), 56),
    ('D2 (1,4,4,8) ST4 cap64', (1, 1, 1, 4, 2, 0, 128, 0, 2304, 8192, 1, 4, 4, 8), 64),
    ('uncapped (1,4,4,8) ST2 NT256 (P1)', (1, 1, 1, 2, 2, 0, 256, 0, 2304, 8192, 1, 4, 4, 8), None),
]
lines = []
for (label, inst, cap) in CASES:
    flags = base + ([f'--maxrregcount={cap}'] if cap else [])
    name = '&v18d_native_gemv<' + ', '.join(str(v) for v in inst) + '>'
    rc, log, cubin, dur = L.NVRTC().compile(dev.encode(), [name], flags, False)
    st = re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", log)
    rg = re.search(r"Used (\d+) registers", log)
    lines.append(f'{label}: rc {rc} regs {rg.group(1) if rg else "?"} stack/spill {st.groups() if st else "?"}')
out = Path('results/p5-offline')
out.mkdir(parents=True, exist_ok=True)
(out / 'summary.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
