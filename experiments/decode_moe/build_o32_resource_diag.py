"""One bounded O32 host-isolation repair; keep the original device source."""

import ast
import base64
import hashlib
import json
import re
import zlib

from pdl16_sources import HERE, ROOT, literal
from build_norm_pair_diag import DENSE_SHA, extract
from build_handoff16_diag import SPIN
from harness import append_harness

OUT = ROOT / 'results/decode-moe-plan-20261030/o32-resource-repair'

QUERY = r'''
// Query-only legacy call, to diagnose configuration sharing. Never launch it.
int o32r_legacy_query() { return ol_launch(0,0,0,0,32,0,1); }
int o32r_attr(int what) {
  auto fn=pdlo32_projection<32,128,8,1,1,1,1>;
  cudaFuncAttributes a{};cudaError_t rc=cudaFuncGetAttributes(&a,fn);
  if(rc!=cudaSuccess)return -1000-int(rc);
  if(what==0)return a.numRegs;
  if(what==1)return int(a.localSizeBytes);
  if(what==2)return int(a.sharedSizeBytes);
  if(what==3)return int(a.maxDynamicSharedSizeBytes);
  if(what==4)return a.maxThreadsPerBlock;
  if(what==5)return a.preferredShmemCarveout;
  if(what==6)return pdlo32_info(1);
  int device=0;rc=cudaGetDevice(&device);if(rc!=cudaSuccess)return -2000-int(rc);
  cudaDeviceProp prop{};rc=cudaGetDeviceProperties(&prop,device);
  if(rc!=cudaSuccess)return -3000-int(rc);
  if(what==7)return int(prop.sharedMemPerMultiprocessor);
  if(what==8)return int(prop.sharedMemPerBlockOptin);
  return -99;
}
'''


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dense = (ROOT / 'best/verified.py').read_bytes()
    assert hashlib.sha256(dense).hexdigest() == DENSE_SHA
    old = (OUT.parent / 'pdl16/o32-candidate.cu').read_bytes().decode()
    assert old.count('int ol_launch_t(') == 1 and old.count('int ol_launch(') == 1
    start = old.index('\nusing OLEncode=')
    middle = old.index('\nint pdlo32_info(', start)
    tail = old.index('\nint ol_launch(', middle)
    host = old[start:middle] + old[tail:]
    private = re.sub(r'\bol_', 'o32r_', host).replace('OLEncode', 'O32REncode')
    inverse = re.sub(r'\bo32r_', 'ol_', private).replace('O32REncode', 'OLEncode')
    assert inverse == host and 'ol_launch_t(' not in private
    cuda = old + private + QUERY + SPIN
    cpp = '''#include <stdint.h>
int o32r_launch(int,int64_t,int64_t,int64_t,int,int64_t,int);
int o32r_legacy_query();
int o32r_attr(int);
int d16_spin_start(int64_t,int64_t,int,int,int64_t,int64_t);
'''
    functions = ['o32r_launch', 'o32r_legacy_query', 'o32r_attr', 'd16_spin_start']
    old_diag = (HERE / 'dense_handoff16_diag.py').read_bytes()
    assert hashlib.sha256(old_diag).hexdigest() == 'b3560cdd3b91cf5a8324f842833e08a61de3e1f72cf64099c17193e91f240415'
    source = old_diag.decode()
    app = append_harness('') + '\nimport hashlib\n'
    app += 'def _o32r_zsrc(s):\n import base64,zlib\n return zlib.decompress(base64.b64decode(s)).decode()\n'
    app += '_O32R_CUDA = _o32r_zsrc(' + repr(base64.b64encode(zlib.compress(cuda.encode(), 9)).decode()) + ')\n'
    for key, value in {'_O32R_CPP': cpp, '_O32R_FUNCTIONS': functions,
                       '_O32R_CUDA_SHA': hashlib.sha256(cuda.encode()).hexdigest()}.items():
        app += key + ' = ' + repr(value) + '\n'
    app += extract(source, ['_r2_find', '_achita_dense_project', '_d16_reference',
                            '_d16_spin', '_d16_times', '_d16_projection_probe',
                            '_pdli_equal', '_pdli_probe'])
    app += (HERE / 'o32_resource_diag_runtime.py').read_text(encoding='utf-8')
    raw = dense + app.encode()
    ast.parse(raw)
    assert len(raw) < 1_000_000 and raw.startswith(dense)
    (HERE / 'dense_o32_resource_diag.py').write_bytes(raw)
    (OUT / 'legacy-device-and-host-prefix.cu').write_bytes(old.encode())
    (OUT / 'private-host.cu').write_bytes(private.encode())
    (OUT / 'diagnostic.cu').write_bytes(cuda.encode())
    (OUT / 'diagnostic.cpp').write_bytes(cpp.encode())
    record = dict(source_sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw),
        Dense_parent_sha256=DENSE_SHA, previous_job='3264689ee9974ef9',
        original_O32_CUDA_sha256=hashlib.sha256(old.encode()).hexdigest(),
        diagnostic_CUDA_sha256=hashlib.sha256(cuda.encode()).hexdigest(),
        old_CUDA_is_exact_prefix=True, private_host_inverse_exact=True,
        device_algorithm_unchanged=True, fixed_family='O32 only, (32,128,8,1,1,1), TW1',
        hypothesis='Generic host template configuration state may be shared across extension modules. Actual H200 cause unconfirmed.',
        correction='Unique host template, map and launch symbols. Legacy query is never launched. Read attributes and occupancy before and after initialization.',
        gates='Positive queried residency, zero local bytes, exact shape/config, finite bytes on changing normal/zero/outlier probes; >3% and >=1.5us in both orders.',
        enables_nothing=True, timed_Dense_parent_unchanged=True, GPU_jobs=0)
    (OUT / 'source-binding.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
