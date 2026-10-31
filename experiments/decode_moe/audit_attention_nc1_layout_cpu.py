"""Compile packaged device entries and constexpr sizes without GPU loading."""
import hashlib
import json
import re
import struct
import subprocess

from attention_nc1_source import OUT, ROOT, derive
from audit_attention_finoutline_cpu import Capture, blocks, nv


class Compiler(Capture):
    def output(self, program, kind):
        if kind == 'CUBIN':
            self.ptx = super().output(program, 'PTX').rstrip(b'\0').decode()
        return super().output(program, kind)


def main():
    _, _, src, _ = derive()
    assert (OUT / 'packaged.cu').read_bytes() == src.encode()
    device = src.split('void launch_nc1(', 1)[0]
    device = re.sub(r'#include <(torch/extension.h|ATen/cuda/CUDAContext.h|type_traits|cuda.h)>\n', '', device)
    device = ('#include <math_constants.h>\n#define INFINITY CUDART_INF_F\n'
              'namespace std { template<class T,T V> struct integral_constant { static constexpr T value=V; }; }\n' + device)
    device += '''
extern "C" __device__ __constant__ unsigned nc1_layout[6] = {
  sizeof(Smem<1,true>)+1024, Cfg<1,true>::NTX, Cfg<1,true>::MINB,
  sizeof(Smem<1,false>)+1024, Cfg<1,false>::NTX, Cfg<1,false>::MINB};
'''
    (OUT / 'packaged-device-reporter.cu').write_bytes(device.encode())
    c = Compiler()
    r = c.compile(device.encode(), ['&attn_kernel_nc1<1,true,0>', '&attn_kernel_nc1<1,false,13>'],
                  [], tags=True, lineinfo=True)
    (OUT / 'packaged-layout.log').write_bytes(r['log'].encode())
    assert r['rc'] == 0, r['log'][-3000:]
    (OUT / 'packaged-layout.ptx').write_bytes(c.ptx.encode())
    binary = OUT / 'packaged-layout.cubin'
    binary.write_bytes(c.cubin)
    match = re.search(r'\.(u32|b8) nc1_layout\[(6|24)\]\s*=\s*\{([^}]+)\}', c.ptx)
    assert match, c.ptx[-3000:]
    values = [int(v.strip()) for v in match.group(3).split(',')]
    values += [0] * (int(match.group(2)) - len(values))
    if match.group(1) == 'b8':
        values = list(struct.unpack('<6I', bytes(values)))
    assert len(values) == 6
    layouts = {name: dict(zip(('dynamic_shared_bytes', 'threads', 'launch_bounds_min_blocks'), values[3*i:3*i+3]))
               for i, name in enumerate(('flat_L1', 'spec_one_L1'))}
    exe = ROOT.parents[2] / '.tools/cuda-compiler-13.0/nvidia/cu13/bin/cuobjdump.exe'
    sass = subprocess.check_output([str(exe), '-sass', str(binary)], text=True)
    (OUT / 'packaged-layout.sass').write_bytes(sass.encode())
    entries = {k: v for k, v in blocks(sass).items() if 'attn_kernel_nc1' in k}
    kernels = nv.parse_ptxas(r['log'])
    assert len(entries) == len(kernels) == 2
    assert all(k['stack_frame'] == k['spill_stores'] == k['spill_loads'] == 0 for k in kernels)
    assert all(not v['local_instructions'] for v in entries.values())
    report = dict(packaged_CUDA_sha256=hashlib.sha256(src.encode()).hexdigest(),
                  layouts=layouts, kernels=kernels, entries=entries, zero_stack_spills=True,
                  GPU_calls=0, admitted_for_GPU=False,
                  limitation='Compile-time sizes/device compilation only;host compile,actual attributes,numerics and timing pending.')
    (OUT / 'packaged-layout-cpu-audit.json').write_bytes((json.dumps(report, indent=2) + '\n').encode())
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
