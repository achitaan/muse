"""Offline device compilation for the fixed L1-wide draft, with layout probes."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from build_attention_l1_wide_tile import OUT, ROOT


def main():
    destination = ROOT / '.tools/attention-l1-wide-preflight-v1'
    destination.mkdir(exist_ok=True)
    names = ['&attn_kernel_fp8pv<1,true,0>', '&attn_kernel_fp8pv<8,true,0>',
             '&attn_kernel_fp8pv<1,false,1>']
    names += [f'&layout_bytes_probe<sizeof(Smem<{l},{flat}>)+1024>'
              for l, flat in [(1, 'true'), (8, 'true'), (1, 'false')]]
    for arm, source_file, defines in [('baseline', 'baseline.cu', []),
                                      ('wide', 'candidate.cu', []),
                                      ('off', 'candidate.cu', ['ATTN_L1_WIDE_TILE=0'])]:
        cuda = (OUT / source_file).read_text()
        anchor = 'template <int L>\nvoid launch('
        assert cuda.count(anchor) == 1
        prefix = cuda.split(anchor)[0]
        removed = ['#include <torch/extension.h>\n', '#include <ATen/cuda/CUDAContext.h>\n', '#include <cuda.h>\n']
        for include in removed:
            assert prefix.count(include) == 1
            prefix = prefix.replace(include, '')
        prefix += '\ntemplate<int N> __global__ void layout_bytes_probe() {}\n'
        prefix += 'static_assert(sizeof(Smem<1,true>) + 1024 <= 115712, "two-block shared-memory screen");\n'
        data = prefix.encode()
        path = destination / (arm + '.cu')
        assert not path.exists() or path.read_bytes() == data, 'Frozen compiler input collision'
        path.write_bytes(data)
        result_path = path.with_suffix('.json')
        if not result_path.exists():
            command = [sys.executable, str(ROOT / 'experiments/prepared/local_nvrtc_compile.py'),
                       '--source', str(path), '--minimal-std-tags', '--out', str(path.with_suffix('')),
                       '--define', 'INFINITY=__int_as_float(0x7f800000)']
            for define in defines:
                command += ['--define', define]
            for name in names:
                command += ['--name', name]
            subprocess.run(command, cwd=ROOT, check=True)
        report = json.loads(result_path.read_text())
        assert report['return_code'] == 0
        assert report['original_source_sha256'] == hashlib.sha256(data).hexdigest()
        print(json.dumps(dict(arm=arm, source_sha256=report['original_source_sha256'],
                              GPU_jobs=0, return_code=report['return_code'])))


if __name__ == '__main__':
    main()
