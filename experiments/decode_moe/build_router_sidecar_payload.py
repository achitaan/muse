"""Package one unscored joint Dense-unit diagnostic; public Dense behavior stays retained."""
import ast
import hashlib
import json
from pathlib import Path
from harness import append_harness
from screen_router_sidecar_cpu import ROOT, OUT, DECODER, derive

DENSE = 'edc61a1e1fb26a10bcfacda3ef3fc06833c17ec7a078d5ded8eb8717509b765b'


def sources():
    parent, candidate = derive()
    renamed = candidate.replace('v18s_gemv', 'sidecar_gemv')
    assert renamed.replace('sidecar_gemv', 'v18s_gemv') == candidate
    bridge = '''
#include <vector>
#include <stdint.h>
std::vector<int64_t> r2_resources() {
  auto kernel = KERNEL<1,1,1,1,0,0,1024,0,4096,8192,2,16,8,1,70>;
  cudaFuncAttributes attr{};
  int blocks = 0, device = 0;
  cudaDeviceProp prop{};
  cudaError_t a = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 16384);
  cudaError_t b = cudaFuncGetAttributes(&attr, kernel);
  cudaError_t c = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernel, 1024, 16384);
  cudaError_t d = cudaGetDevice(&device);
  cudaError_t e = cudaGetDeviceProperties(&prop, device);
  if (a != cudaSuccess || b != cudaSuccess || c != cudaSuccess || d != cudaSuccess || e != cudaSuccess) {
    cudaGetLastError();
    return {-1, (int64_t)a, (int64_t)b, (int64_t)c, (int64_t)d, (int64_t)e};
  }
  return {attr.numRegs, (int64_t)attr.localSizeBytes, (int64_t)attr.sharedSizeBytes,
          16384, blocks, prop.multiProcessorCount};
}
'''
    raw = (ROOT / 'experiments/decode_moe/decoder_pdli.py').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == DECODER
    node = next(n for n in ast.parse(raw).body if isinstance(n, ast.Assign)
                and getattr(n.targets[0], 'id', '') == '_R2_CPP')
    cpp = ast.literal_eval(node.value) + '''
#include <pybind11/stl.h>
#include <vector>
std::vector<int64_t> r2_resources();
'''
    return parent + bridge.replace('KERNEL', 'v18s_gemv'), renamed + bridge.replace('KERNEL', 'sidecar_gemv'), cpp


def main():
    raw = (ROOT / 'best/verified.py').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == DENSE
    audit = json.loads((OUT / 'cpu-audit.json').read_text())
    assert audit['clean_resource_screen'] and not audit['admitted_for_GPU']
    dep = json.loads((OUT / 'dependency-audit.json').read_text())
    assert dep['candidate_CUDA_sha256'] == audit['candidate_CUDA_sha256']
    parent, candidate, cpp = sources()
    (OUT / 'parent-packaged.cu').write_bytes(parent.encode())
    (OUT / 'candidate-packaged.cu').write_bytes(candidate.encode())
    (OUT / 'diagnostic.cpp').write_bytes(cpp.encode())
    runtime = (ROOT / 'experiments/decode_moe/router_sidecar_runtime.py').read_text()
    payload = append_harness(raw.decode()) + '\n'
    payload += '_rs_parent_prepare = prepare\n_rs_parent_kernel_fn = kernel_fn\n'
    payload += 'RS_PARENT_CUDA = ' + repr(parent) + '\nRS_CANDIDATE_CUDA = ' + repr(candidate) + '\nRS_CPP = ' + repr(cpp) + '\n'
    payload += runtime
    encoded = payload.encode()
    ast.parse(encoded)
    assert len(encoded) < 1_000_000 and encoded.startswith(raw.rstrip(b'\n'))
    path = ROOT / 'experiments/decode_moe/dense_router_sidecar_diagnostic.py'
    path.write_bytes(encoded)
    record = dict(source_sha256=hashlib.sha256(encoded).hexdigest(), bytes=len(encoded),
                  Dense_parent_sha256=DENSE, Decode_parent_sha256=DECODER,
                  CUDA_parent_sha256=hashlib.sha256(parent.encode()).hexdigest(),
                  CUDA_candidate_sha256=hashlib.sha256(candidate.encode()).hexdigest(),
                  screened_device_candidate_sha256=audit['candidate_CUDA_sha256'],
                  original_public_Dense_prefix_exact=True,
                  scope='Unscored preparation-time joint pooler/router diagnostic with actual Dense weight bank;public Dense kernel always retained.',
                  actual_private_marker='sidecar_gemv',
                  numerical_gate='Pooler output,FP32 partials,router weights,router IDs and counter byte-identical to retained R1;unchanged-parent control mandatory.12 changing probes acrossM1,8,2,4.',
                  performance_gate='M1 only;two populations,two arm orders,25 observations per arm,strict candidate<0.97*parent;then two unchanged-parent controls within2%;raw event observations logged.',
                  preparation_limits={'before_probes_seconds':268,'before_marker_seconds':274,'before_timings_seconds':278},
                  required_next='CPU packaged-device/ABI/public-fallback audit and fresh admission;no GPU launch authorized by this record alone.',
                  no_Dense_gain_claim=True, no_Decode_transfer=True, no_scored_submission=True,
                  GPU_jobs=0, admitted_for_GPU=False)
    (OUT / 'payload-binding.json').write_bytes((json.dumps(record, indent=2) + '\n').encode())
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
