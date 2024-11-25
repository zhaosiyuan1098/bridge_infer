#include "buffer.h"

Buffer::Buffer(size_t byte_size, std::shared_ptr<Allocator> allocator, void *ptr,
               bool is_admin)
    : byte_size_(byte_size),
      allocator_(allocator),
      ptr_(ptr),
      is_admin(is_admin)
{
    if (!ptr_ && allocator_)
    {
        device_type_ = allocator_->device_type();
        is_admin = true;
        ptr_ = allocator_->allocate(byte_size);
    }
}

bool Buffer::allocate()
{
    if (allocator_ && byte_size_ != 0)
    {
        is_admin = true;
        ptr_ = allocator_->allocate(byte_size_);
        if (!ptr_)
        {
            return false;
        }
        else
        {
            return true;
        }
    }
    else
    {
        return false;
    }
}

Buffer::~Buffer()
{
    if (is_admin)
    {
        if (ptr_ && allocator_)
        {
            allocator_->release(ptr_);
            ptr_ = nullptr;
        }
    }
}


