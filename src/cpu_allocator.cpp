#include "allocator.h"


void *CPU_Allocator::allocate(size_t byte_size) const
{
    if (!byte_size)
    {
        return nullptr;
    }
    void *data = malloc(byte_size);
    return data;
}

void CPU_Allocator::release(void* ptr) const {
  // 实现代码
}


