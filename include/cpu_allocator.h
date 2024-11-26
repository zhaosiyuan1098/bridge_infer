#ifndef CPU_ALLOCATOR_H
#define CPU_ALLOCATOR_H

#include"allocator.h"

class CPU_Allocator : public Allocator {
 public:
  explicit CPU_Allocator();

  void* allocate(size_t byte_size) const override;

  void release(void* ptr) const override;
};

#endif