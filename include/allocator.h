#ifndef ALLOCATOR_H
#define ALLOCATOR_H
#include"common.h"


// Assuming DeviceType is an enum, you need to define it

class Allocator
{
private:
  /* data */
  DeviceType device_type_ = DeviceType::kDeviceUnknown;

public:
  explicit Allocator(DeviceType device_type) : device_type_(device_type) {}
  virtual DeviceType device_type() const { return device_type_; }
  virtual void release(void *ptr)const = 0;
  virtual void *allocate(size_t byte_size) const = 0;
  virtual void memcpy(const void *src_ptr, void *dest_ptr, size_t byte_size,
                      MemcpyKind memcpy_kind = MemcpyKind::kMemcpyCPU2CPU, void *stream = nullptr,
                      bool need_sync = false) const;
  virtual void memset_zero(void *ptr, size_t byte_size, void *stream, bool need_sync = false);
};


class CPU_Allocator : public Allocator {
 public:
  explicit CPU_Allocator();

  void* allocate(size_t byte_size) const override;

  void release(void* ptr) const override;
};

struct CudaMemoryBuffer {
  void* data;
  size_t byte_size;
  bool busy;

  CudaMemoryBuffer() = default;

  CudaMemoryBuffer(void* data, size_t byte_size, bool busy)
      : data(data), byte_size(byte_size), busy(busy) {}
};

class CUDA_Allocator : public Allocator {
 public:
  explicit CUDA_Allocator();

  void* allocate(size_t byte_size) const override;

  void release(void* ptr) const override;

 private:
  mutable std::map<int, size_t> no_busy_cnt_;
  mutable std::map<int, std::vector<CudaMemoryBuffer>> big_buffers_map_;
  mutable std::map<int, std::vector<CudaMemoryBuffer>> cuda_buffers_map_;
};

#endif