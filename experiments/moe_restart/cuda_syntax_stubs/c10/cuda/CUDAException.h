#pragma once
#define C10_CUDA_CHECK(call) do { (void)(call); } while (0)
#define C10_CUDA_KERNEL_LAUNCH_CHECK() do {} while (0)
