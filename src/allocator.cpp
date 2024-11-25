#include "allocator.h"

void Allocator::memcpy(const void *src_ptr, void *dest_ptr, size_t byte_size,
                       MemcpyKind memcpy_kind, void *stream, bool need_sync) const
{
  CHECK_NE(src_ptr, nullptr);
  CHECK_NE(dest_ptr, nullptr);
  if (!byte_size)
  {
    return;
  }

  cudaStream_t stream_ = nullptr;
  if (stream)
  {
    stream_ = static_cast<CUstream_st *>(stream);
  }

  switch (memcpy_kind)
  {
  case MemcpyKind::kMemcpyCPU2CPU:
    std::memcpy(dest_ptr, src_ptr, byte_size);
    break;
  case MemcpyKind::kMemcpyCPU2CUDA:
    if (!stream_)
    {
      cudaMemcpy(dest_ptr, src_ptr, byte_size, cudaMemcpyHostToDevice);
    }
    else
    {
      cudaMemcpyAsync(dest_ptr, src_ptr, byte_size, cudaMemcpyHostToDevice, stream_);
    }
    break;
  case MemcpyKind::kMemcpyCUDA2CPU:
    if (!stream_)
    {
      cudaMemcpy(dest_ptr, src_ptr, byte_size, cudaMemcpyDeviceToHost);
    }
    else
    {
      cudaMemcpyAsync(dest_ptr, src_ptr, byte_size, cudaMemcpyDeviceToHost, stream_);
    }
    break;
  case MemcpyKind::kMemcpyCUDA2CUDA:
    if (!stream_)
    {
      cudaMemcpy(dest_ptr, src_ptr, byte_size, cudaMemcpyDeviceToDevice);
    }
    else
    {
      cudaMemcpyAsync(dest_ptr, src_ptr, byte_size, cudaMemcpyDeviceToDevice, stream_);
    }
    break;
  default:
    std::cout << "Unknown memcpy kind: " << int(memcpy_kind);
    break;
  }

  if (need_sync)
  {
    cudaDeviceSynchronize();
  }
  cudaStreamDestroy(stream_);
}

void Allocator::memset_zero(void *ptr, size_t byte_size, void *stream, bool need_sync)
{
  std::cout << "todo" << std::endl;
}
