# Helper-local alternative adapted from frozen grouped-norm tooling.
"""Compile exported constexpr layout data; never load/launch a CUDA module."""
import hashlib
import json
import re
import struct
import sys

from norm_register_source import OUT, derive
from tinyroute_source import HERE, ROOT


def main():
    _, source, _, values = derive()
    assert (OUT/'packaged.cu').read_bytes() == source.encode()
    sys.path.insert(0, str(HERE.parents[2]/'opus-fresh/experiments/opus_fresh'))
    import nvrtc_screen_alternatives as nv

    class Compiler(nv.NVRTC):
        def output(self, program, kind):
            if kind == 'CUBIN':
                self.ptx = super().output(program, 'PTX').rstrip(b'\0').decode()
            return super().output(program, kind)

    # Evaluate the retained host allocation expression using device-side
    # constexpr values of the exact structure, not a Python field-size model.
    reporter = '''
using namespace norm_reg;
constexpr unsigned layout_base = ((sizeof(Smem) + 127) / 128) * 128;
constexpr unsigned layout_xa = OKC_NS * UNIT_BYTES + (OKC_SKT > 8 ? NW * 8 * X1P * 2 : 0);
constexpr unsigned layout_xb = NSF * UNIT_BYTES + 2 * FT_T * XP_FT * 2;
extern "C" __device__ __constant__ unsigned okc_layout_report[7] = {
  sizeof(Smem),layout_base,layout_xa,layout_xb,
  layout_base + (layout_xa > layout_xb ? layout_xa : layout_xb),NT,NTL};
'''
    device = source.split('#include <torch/extension.h>', 1)[0] + reporter
    (OUT/'layout-reporter.cu').write_bytes(device.encode())
    compiler = Compiler()
    result = {}
    for name in ('TINY', 'TINY2'):
        defines = list(values[name+'_DEFINES']) + ['-DOKC_DQ7=1','-DOKC_NS=3','-DOKC_NSF=3','-DOKC_PRMTSC=1']
        compiled = compiler.compile(device.encode(), [], defines, tags=True, lineinfo=False)
        (OUT/(name+'-layout.log')).write_bytes(compiled['log'].encode())
        assert compiled['rc'] == 0, compiled['log']
        (OUT/(name+'-layout.ptx')).write_bytes(compiler.ptx.encode())
        match = re.search(r'\.(u32|b8) okc_layout_report\[(7|28)\]\s*=\s*\{([^}]+)\}', compiler.ptx)
        assert match, compiler.ptx[-3000:]
        data = [int(v.strip()) for v in match.group(3).split(',')]
        # PTX aggregate initializers may omit trailing zero elements.
        assert len(data) <= int(match.group(2))
        data += [0] * (int(match.group(2)) - len(data))
        numbers = data if match.group(1) == 'u32' else list(struct.unpack('<7I', bytes(data)))
        assert len(numbers) == 7 and numbers[-2:] == [256,288]
        result[name] = dict(zip(('Smem_bytes','aligned_base_bytes','phase_A_bytes','phase_B_bytes','dynamic_shared_bytes','compute_threads','launch_threads'), numbers))
        result[name]['layout_PTX_sha256'] = hashlib.sha256(compiler.ptx.encode()).hexdigest()
        result[name]['reporter_CUDA_sha256'] = hashlib.sha256(device.encode()).hexdigest()
    header = ROOT.parents[2]/'.tools/cuda-compiler-13.0/nvidia/cu13/include/cuda_occupancy.h'
    report = dict(candidate_CUDA_sha256=hashlib.sha256(source.encode()).hexdigest(),
                  layouts=result,occupancy_header_sha256=hashlib.sha256(header.read_bytes()).hexdigest(),
                  GPU_calls=0,limitation='Compile-time sizes only; no actual GPU function attributes or residency query.')
    (OUT/'layout-cpu-audit.json').write_bytes((json.dumps(report,indent=2)+'\n').encode())
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
