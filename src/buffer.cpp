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

void Buffer::copy_from(const Buffer &buffer) const
{
    CHECK(allocator_ != nullptr);
    CHECK(buffer.ptr_ != nullptr);

    size_t byte_size = byte_size_ < buffer.byte_size_ ? byte_size_ : buffer.byte_size_;
    const DeviceType &buffer_device = buffer.device_type();
    const DeviceType &current_device = this->device_type();
    CHECK(buffer_device != DeviceType::kDeviceUnknown &&
          current_device != DeviceType::kDeviceUnknown);

    MemcpyKind memcpy_kind;
    switch (buffer_device)
    {
    case DeviceType::kDeviceCPU:
        switch (current_device)
        {
        case DeviceType::kDeviceCPU:
            memcpy_kind = MemcpyKind::kMemcpyCPU2CPU;
            break;
        case DeviceType::kDeviceCUDA:
            memcpy_kind = MemcpyKind::kMemcpyCPU2CUDA;
            break;
        default:
            std::cout << "Unsupported device type";
        }
        break;
    case DeviceType::kDeviceCUDA:
        switch (current_device)
        {
        case DeviceType::kDeviceCPU:
            memcpy_kind = MemcpyKind::kMemcpyCUDA2CPU;
            break;
        case DeviceType::kDeviceCUDA:
            memcpy_kind = MemcpyKind::kMemcpyCUDA2CUDA;
            break;
        default:
            std::cout << "Unsupported device type";
        }
        break;
    default:
        std::cout << "Unsupported device type";
    }

    allocator_->memcpy(buffer.ptr(), this->ptr_, byte_size, memcpy_kind);
}

void Buffer::copy_from(const Buffer *buffer) const
{
    CHECK(allocator_ != nullptr);
    CHECK(buffer != nullptr || buffer->ptr_ != nullptr);

    size_t src_size = byte_size_;
    size_t dest_size = buffer->byte_size_;
    size_t byte_size = src_size < dest_size ? src_size : dest_size;

    const DeviceType &buffer_device = buffer->device_type();
    const DeviceType &current_device = this->device_type();
    CHECK(buffer_device != DeviceType::kDeviceUnknown &&
          current_device != DeviceType::kDeviceUnknown);

    MemcpyKind memcpy_kind;
    switch (buffer_device)
    {
    case DeviceType::kDeviceCPU:
        switch (current_device)
        {
        case DeviceType::kDeviceCPU:
            memcpy_kind = MemcpyKind::kMemcpyCPU2CPU;
            break;
        case DeviceType::kDeviceCUDA:
            memcpy_kind = MemcpyKind::kMemcpyCPU2CUDA;
            break;
        default:
            std::cout << "Unsupported device type";
            return;
        }
        break;
    case DeviceType::kDeviceCUDA:
        switch (current_device)
        {
        case DeviceType::kDeviceCPU:
            memcpy_kind = MemcpyKind::kMemcpyCUDA2CPU;
            break;
        case DeviceType::kDeviceCUDA:
            memcpy_kind = MemcpyKind::kMemcpyCUDA2CUDA;
            break;
        default:
            std::cout << "Unsupported device type";
            return;
        }
        break;
    default:
        std::cout << "Unsupported device type";
        return;
    }

    allocator_->memcpy(buffer->ptr_, this->ptr_, byte_size, memcpy_kind);
}
