"""Device-only NVRTC/ptxas, never loads a CUDA driver or launches a kernel."""
import os, sys, json, subprocess, hashlib
from pathlib import Path
os.environ['CUDA_CACHE_DISABLE'] = '1'
sys.dont_write_bytecode = True
from build_source import ROOT, source
sys.path.insert(0, str(ROOT/'experiments/prepared'))
sys.path.insert(0, str(ROOT.parents[2]/'experiments/prepared'))
import local_nvrtc_compile as nv
flags = ['--gpu-architecture=sm_90a', '--std=c++17', '--generate-line-info',
         '--device-as-default-execution-space', '--ptxas-options=-v',
         '--include-path='+str(nv.TOOL/'nvidia/cu13/include'),
         '--include-path='+str(nv.TOOL/'nvidia/cu13/include/cccl')]
cu = source().split('#include <torch/extension.h>')[0]
out = ROOT/'.fs_offline'
(out/'device.cu').write_text(cu)
rc, log, cubin, secs = nv.NVRTC().compile(cu.encode(), ['&fs::kernel<false>', '&fs::kernel<true>'], flags, True)
(out/'compile.log').write_text(log)
print('FS_COMPILE', rc, secs, log, flush=True)
if rc: raise SystemExit(rc)
(out/'fs.cubin').write_bytes(cubin)
for mode, ext in [('--dump-resource-usage', 'resources'), ('--dump-sass', 'sass')]:
    (out/('fs.'+ext)).write_text(subprocess.check_output([str(nv.TOOL/'nvidia/cu13/bin/cuobjdump.exe'), mode, str(out/'fs.cubin')], text=True))
(out/'compile.json').write_text(json.dumps(dict(rc=rc, seconds=secs, flags=flags, device_sha256=hashlib.sha256(cu.encode()).hexdigest()), indent=2))
