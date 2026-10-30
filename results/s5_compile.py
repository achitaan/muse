import ast
import base64
import re
import sys
import zlib
from pathlib import Path

sys.path.insert(0, r"C:\Users\Achita\Documents\ChatGPT\Muse 2\experiments\prepared")
import local_nvrtc_compile as L

flags = ['--gpu-architecture=sm_90a', '--std=c++20', '--device-as-default-execution-space', '--ptxas-options=-v',
         '--include-path=' + str(L.TOOL / 'nvidia/cu13/include'),
         '--include-path=' + str(L.TOOL / 'nvidia/cu13/include/cccl')]
lines = []


def summarize(tag, rc, log):
    lines.append(f'{tag}: rc {rc}')
    for b in re.split(r"(?=ptxas info\s*: Compiling entry function)", log):
        st = re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", b)
        rg = re.search(r"Used (\d+) registers", b)
        nm = re.search(r"entry function '([^']+)'", b)
        if st and rg:
            lines.append(f'  {nm.group(1)[:70] if nm else "?"} regs {rg.group(1)} stack/spill {st.groups()}')
    if rc:
        lines.append(log[:1500])


t = ast.parse(Path('best/verified.py').read_text(encoding='utf-8'))
rw = None
for n in t.body:
    if isinstance(n, ast.Assign) and getattr(n.targets[0], 'id', '') == 'rw_SOURCES':
        for k, v in zip(n.value.keys, n.value.values):
            if k.value == 128:
                rw = zlib.decompress(base64.b64decode(v.args[0].value)).decode()
ns = {}
exec(Path('experiments/dense_s5.py').read_text(encoding='utf-8').split('# --- runtime ---')[0], ns)
src = ns['_s5_rw_source'](rw)
dev = src.split('typedef CUresult')[0]
for inc in ('#include <cuda.h>\n', '#include <cuda_runtime.h>\n', '#include <stdint.h>\n', '#include <algorithm>\n'):
    dev = dev.replace(inc, '')
dev = ('#include <cuda/std/cstdint>\nusing namespace cuda::std;\n'
       'struct alignas(64) CUtensorMap_st { unsigned long long opaque[16]; }; typedef CUtensorMap_st CUtensorMap;\n' + dev)
names = [f'&s5rw128_kernel<{bn},{s},8,1,2,1>' for (bn, s) in ns['_S5_RW_VARIANTS']]
rc, log, cubin, dur = L.NVRTC().compile(dev.encode(), names, flags, False)
summarize('router', rc, log)

tpl = Path(r'C:\Users\Achita\AppData\Local\Temp\opencode\v18s_template.cu').read_text(encoding='utf-8')
dv = tpl.split('static int v18s_smem_optin()')[0]
for inc in ('#include <cuda_runtime.h>\n', '#include <algorithm>\n', '#include <stdint.h>\n', '#include <type_traits>\n'):
    dv = dv.replace(inc, '')
dv = ('#include <cuda/std/cstdint>\n#include <cuda/std/type_traits>\nusing namespace cuda::std;\n'
      'namespace std { using cuda::std::conditional; }\n' + dv)
insts = []
for m in ns['_S5_O_MS']:
    for (wr, wk, ku, s) in ns['_S5_O_GEOMS']:
        insts.append((1, (m + 7) // 8, 1, 2, 0, 0, wr * wk * 32, 0, 8192, 1024, wr, wk, ku, s, 6))
rc, log, cubin, dur = L.NVRTC().compile(dv.encode(), ['&v18s_gemv<' + ', '.join(str(v) for v in i) + '>' for i in insts], flags, False)
summarize('o_v18s', rc, log)
out = Path('results/s5-offline')
out.mkdir(parents=True, exist_ok=True)
(out / 'summary.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
