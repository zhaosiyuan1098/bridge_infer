#include"allocator.h"
#include <cassert>


void* CUDA_Allocator::allocate(size_t byte_size) const {
  if (!byte_size) {
    return nullptr;
  }
  void* ptr = nullptr;
  cudaError_t err = cudaMalloc(&ptr, byte_size);
  CHECK_EQ(err, cudaSuccess);
  return ptr;
}

void CUDA_Allocator::release(void* ptr) const {
  // 实现代码
}