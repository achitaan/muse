"""Complete the native host interface; no upload, CUDA load or kernel call."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'STATE.json').exists())

HOST_HELPERS = r'''
static bool g_cluster2_ready = false;
static int64_t g_cluster2_grid = 0;
static int g_cluster2_device = -1;
static void cluster2_check(cudaError_t status, const char* operation) {
  TORCH_CHECK(status == cudaSuccess, operation, ": ", cudaGetErrorString(status));
}
static void cluster2_config(cudaLaunchConfig_t& cfg, cudaLaunchAttribute (&attrs)[2],
                            int64_t grid, size_t sm, cudaStream_t stream) {
  cfg = {};
  cfg.gridDim = dim3((unsigned)grid);
  cfg.blockDim = dim3(okc_cluster2::NTL);
  cfg.dynamicSmemBytes = sm;
  cfg.stream = stream;
  attrs[0] = {};
  attrs[0].id = cudaLaunchAttributeClusterDimension;
  attrs[0].val.clusterDim.x = 2;
  attrs[0].val.clusterDim.y = 1;
  attrs[0].val.clusterDim.z = 1;
  attrs[1] = {};
  attrs[1].id = cudaLaunchAttributeCooperative;
  attrs[1].val.cooperative = 1;
  cfg.attrs = attrs;
  cfg.numAttrs = 2;
}
std::vector<int64_t> okc_cluster2_info(int64_t grid) {
  g_cluster2_ready = false;
  TORCH_CHECK(grid > 0 && grid <= okc_cluster2::WGS_SLOTS && (grid & 1) == 0,
              "Cluster2 requires an even grid within scratch capacity");
  int dev = -1, cluster_support = 0, cooperative_support = 0;
  cluster2_check(cudaGetDevice(&dev), "get cluster device");
  cluster2_check(cudaDeviceGetAttribute(&cluster_support, cudaDevAttrClusterLaunch, dev), "cluster support");
  cluster2_check(cudaDeviceGetAttribute(&cooperative_support, cudaDevAttrCooperativeLaunch, dev), "cooperative support");
  const size_t sm = okc_smem_bytes();
  cluster2_check(cudaFuncSetAttribute(okc_cluster2::moe_fused<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm), "small dynamic shared memory");
  cluster2_check(cudaFuncSetAttribute(okc_cluster2::moe_fused<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm), "big dynamic shared memory");
  cudaFuncAttributes a{}, b{};
  cluster2_check(cudaFuncGetAttributes(&a, okc_cluster2::moe_fused<false>), "small attributes");
  cluster2_check(cudaFuncGetAttributes(&b, okc_cluster2::moe_fused<true>), "big attributes");
  int ca = 0, cb = 0;
  if (cluster_support && cooperative_support) {
    cudaLaunchConfig_t cfg{};
    cudaLaunchAttribute attrs[2]{};
    cluster2_config(cfg, attrs, grid, sm, nullptr);
    cluster2_check(cudaOccupancyMaxActiveClusters(&ca, okc_cluster2::moe_fused<false>, &cfg), "small active clusters");
    cluster2_check(cudaOccupancyMaxActiveClusters(&cb, okc_cluster2::moe_fused<true>, &cfg), "big active clusters");
  }
  const bool fields_ok = a.numRegs > 0 && a.numRegs <= 112 && b.numRegs > 0 && b.numRegs <= 112 &&
      a.localSizeBytes == 0 && b.localSizeBytes == 0 && a.sharedSizeBytes == 0 && b.sharedSizeBytes == 0 &&
      sm == 231168 && okc_cluster2::NTL == 576 && a.maxThreadsPerBlock >= okc_cluster2::NTL &&
      b.maxThreadsPerBlock >= okc_cluster2::NTL && a.requiredClusterWidth == 2 && a.requiredClusterHeight == 1 &&
      a.requiredClusterDepth == 1 && b.requiredClusterWidth == 2 && b.requiredClusterHeight == 1 && b.requiredClusterDepth == 1;
  g_cluster2_ready = fields_ok && cluster_support && cooperative_support && ca >= grid / 2 && cb >= grid / 2;
  g_cluster2_grid = grid;
  g_cluster2_device = dev;
  return {(int64_t)a.numRegs, (int64_t)b.numRegs, (int64_t)a.localSizeBytes, (int64_t)b.localSizeBytes,
          (int64_t)a.sharedSizeBytes, (int64_t)b.sharedSizeBytes, (int64_t)sm, (int64_t)okc_cluster2::NTL,
          cluster_support, cooperative_support, ca, cb,
          a.requiredClusterWidth, a.requiredClusterHeight, a.requiredClusterDepth,
          b.requiredClusterWidth, b.requiredClusterHeight, b.requiredClusterDepth,
          grid, (int64_t)g_cluster2_ready, 20261029};
}
'''

LAUNCH = r'''  TORCH_CHECK(g_cluster2_ready && grid == g_cluster2_grid,
              "Cluster2 was not admitted during prepare");
  TORCH_CHECK(g_okc_pdl == 0, "Cluster2 uses a cooperative launch with PDL disabled");
  int current_device = -1;
  cluster2_check(cudaGetDevice(&current_device), "current cluster device");
  TORCH_CHECK(current_device == g_cluster2_device, "Cluster2 device changed after prepare");
  const size_t sm = okc_smem_bytes();
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  cudaLaunchConfig_t cfg{};
  cudaLaunchAttribute attrs[2]{};
  cluster2_config(cfg, attrs, grid, sm, st);
  if (p.M >= OKC_BIGM)
    cluster2_check(cudaLaunchKernelEx(&cfg, okc_cluster2::moe_fused<true>, p), "launch big cluster2");
  else
    cluster2_check(cudaLaunchKernelEx(&cfg, okc_cluster2::moe_fused<false>, p), "launch small cluster2");
}
'''


def main():
    raw = (ROOT / 'best/moe_verified.py').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == '331148716e8d1ba92b45cd29725be997d44b3a5235af74d75770bb9444100d50'
    tree = ast.parse(raw)
    original = next(n.value.value for n in tree.body if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == 'CUDA_SRC' for t in n.targets))
    boundary = '#include <torch/extension.h>'
    parent_prefix, host = original.split(boundary, 1)
    host = boundary + host
    out = ROOT / '.tools/moe-cluster-x'
    device = (out / 'device.cu').read_bytes()
    audit = json.loads((out / 'device-audit.json').read_bytes())
    assert hashlib.sha256(parent_prefix.encode()).hexdigest() == audit['parent_device_prefix_sha256']
    assert hashlib.sha256(device).hexdigest() == audit['device_prefix_sha256']
    old_host = host
    edits = []

    def edit(name, old, new, count=1):
        nonlocal host
        assert host.count(old) == count, (name, host.count(old), count)
        host = host.replace(old, new)
        edits.append(dict(name=name, old=old, new=new, count=count))

    edit('host namespace binding', 'okc::', 'okc_cluster2::', host.count('okc::'))
    anchor = 'void okc_set_pdl(int64_t on)'
    edit('prepared resource interface', anchor, HOST_HELPERS + '\n' + anchor)
    start = host.index('  static bool attr = false;')
    old = host[start:]
    assert old.rstrip().endswith('}')
    edit('all launches cooperative cluster2', old, LAUNCH)
    inverse = host
    for item in reversed(edits):
        assert inverse.count(item['new']) == item['count'], item['name']
        inverse = inverse.replace(item['new'], item['old'])
    assert inverse == old_host
    full = device + host.encode()
    assert full.split(boundary.encode(), 1)[0] == device
    assert b'<<<' not in host.encode(), 'No plain launch fallback is permitted'
    assert host.count('cudaLaunchKernelEx(') == 2 and host.count('cudaLaunchAttributeCooperative') == 1
    assert host.count('okc_cluster2::moe_fused<false>') == 4 and host.count('okc_cluster2::moe_fused<true>') == 4
    (out / 'full.cu').write_bytes(full)
    (out / 'host-edits.json').write_text(json.dumps(edits, indent=2) + '\n', encoding='utf-8')
    report = dict(full_CUDA_sha256=hashlib.sha256(full).hexdigest(),
                  device_prefix_sha256=hashlib.sha256(device).hexdigest(),
                  device_matches_compiled_prefix_before_NVRTC_forward_declaration=True,
                  full_bytes=len(full), host_inverse_recovers_parent=True,
                  original_pointer_and_scratch_mapping_preserved=True,
                  launch_attributes=['ClusterDimension=2,1,1', 'Cooperative=1'],
                  no_plain_launch_fallback=True, PDL_disabled=True,
                  attributes_and_cluster_capacity_required_in_prepare=True,
                  query='okc_cluster2_info(int64_t grid)', resource_fields=21,
                  host_compiled_on_actual_NVCC=False, kernel_execution=False, GPU_jobs=0)
    (out / 'host-audit.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
