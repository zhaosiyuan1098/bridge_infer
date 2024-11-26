#include "cuda_allocator.h"

CUDA_Allocator::CUDA_Allocator() : Allocator(DeviceType::kDeviceCUDA) {}

void *CUDA_Allocator::allocate(size_t byte_size) const
{
  if (!byte_size)
  {
    return nullptr;
  }
  void *ptr = nullptr;
  cudaError_t err = cudaMalloc(&ptr, byte_size);
  CHECK_EQ(err, cudaSuccess);
  return ptr;
}

void CUDA_Allocator::release(void *ptr) const
{
  if (ptr)
  {
    cudaError_t err = cudaFree(ptr);
    CHECK_EQ(err, cudaSuccess);
  }
}