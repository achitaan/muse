#pragma once
#include <cuda_runtime.h>
namespace at { namespace cuda {
cudaStream_t getCurrentCUDAStream();
cudaDeviceProp* getCurrentDeviceProperties();
}}
