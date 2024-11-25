#include "allocator.h"

void Allocator::memcpy(const void* src_ptr, void* dest_ptr, size_t byte_size,
                             MemcpyKind memcpy_kind, void* stream, bool need_sync) const {
  std::cout<<"todo"<<std::endl;
}

void Allocator::memset_zero(void* ptr, size_t byte_size, void* stream, bool need_sync) {
  std::cout << "todo" << std::endl;
}
