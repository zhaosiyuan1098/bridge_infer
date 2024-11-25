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

    void copy_from(const Buffer &buffer) const;
};


#endif