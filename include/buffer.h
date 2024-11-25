#ifndef BUFFER_H
#define BUFFER_H
#include "common.h"
#include "allocator.h"
class Buffer
{
private:
    size_t byte_size_ = 0;
    void *ptr_ = nullptr;
    bool is_admin = true;
    DeviceType device_type_ = DeviceType::kDeviceUnknown;
    std::shared_ptr<Allocator> allocator_;

public:
    explicit Buffer() = default;

    explicit Buffer(size_t byte_size, std::shared_ptr<Allocator> allocator = nullptr,
                    void *ptr = nullptr, bool use_external = false);

    virtual ~Buffer();

    bool allocate();

    DeviceType device_type() const { return device_type_; }
    void *ptr() const { return ptr_; }

    size_t byte_size() const { return byte_size_; }
    void copy_from(const Buffer *buffer) const;
    void copy_from(const Buffer &buffer) const;
};


#endif