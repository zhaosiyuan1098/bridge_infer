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


#endif