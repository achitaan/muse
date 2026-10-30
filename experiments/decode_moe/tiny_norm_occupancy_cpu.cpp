// CPU-only use of NVIDIA's runtime-independent CUDA 13.0 occupancy header.
// Hopper capacities are from NVIDIA's Hopper Tuning Guide; these are model
// inputs, not properties newly queried from a judge GPU.
#include <cuda_occupancy.h>
#include <cstdio>
#include <cstdlib>

int main(int argc, char** argv) {
  if (argc != 3) return 2;
  const int registers = std::atoi(argv[1]);
  const size_t dynamic = std::strtoull(argv[2], nullptr, 10);
  cudaOccDeviceProp device;
  device.computeMajor = 9;
  device.computeMinor = 0;
  device.maxThreadsPerBlock = 1024;
  device.maxThreadsPerMultiprocessor = 2048;
  device.regsPerBlock = device.regsPerMultiprocessor = 65536;
  device.warpSize = 32;
  device.sharedMemPerBlock = 49152;
  device.sharedMemPerMultiprocessor = 228 * 1024;
  device.sharedMemPerBlockOptin = 227 * 1024;
  device.reservedSharedMemPerBlock = 1024;
  device.numSms = 132;
  cudaOccFuncAttributes function;
  function.maxThreadsPerBlock = 288;
  function.numRegs = registers;
  function.sharedSizeBytes = 0;
  function.shmemLimitConfig = FUNC_SHMEM_LIMIT_OPTIN;
  function.maxDynamicSharedSizeBytes = dynamic;
  function.numBlockBarriers = 3;
  cudaOccDeviceState state;
  cudaOccResult result{};
  const auto status = cudaOccMaxActiveBlocksPerMultiprocessor(
      &result, &device, &function, &state, 288, dynamic);
  std::printf("{\"status\":%d,\"active_blocks_per_SM\":%d,"
              "\"registers_per_thread\":%d,\"dynamic_shared_bytes\":%llu,"
              "\"allocated_registers_per_block\":%d,"
              "\"allocated_shared_bytes_per_block\":%llu,"
              "\"block_limit_registers\":%d,\"block_limit_shared\":%d,"
              "\"block_limit_warps\":%d,\"block_limit_barriers\":%d}\n",
              int(status), result.activeBlocksPerMultiprocessor, registers,
              static_cast<unsigned long long>(dynamic),
              result.allocatedRegistersPerBlock,
              static_cast<unsigned long long>(result.allocatedSharedMemPerBlock),
              result.blockLimitRegs, result.blockLimitSharedMem,
              result.blockLimitWarps, result.blockLimitBarriers);
  return status == CUDA_OCC_SUCCESS ? 0 : 1;
}
