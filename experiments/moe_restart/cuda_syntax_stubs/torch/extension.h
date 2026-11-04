// Declaration-only, validation-only Torch surface. Never used by the candidate.
#pragma once
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdlib.h>
#include <vector>
namespace torch {
struct Tensor {
  int device() const;
  int64_t size(int) const;
  void* data_ptr() const;
  template<class T> T* data_ptr() const;
};
}
#define TORCH_CHECK(condition, ...) do { if (!(condition)) __builtin_trap(); } while (0)
