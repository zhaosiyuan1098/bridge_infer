#include "common.h"
#include "allocator.h"
#include "buffer.h"
namespace tensor
{
    class Tensor
    {
    public:
        explicit Tensor() = default;

        explicit Tensor(DataType data_type, int32_t dim0, bool need_alloc = false,
                        std::shared_ptr<Allocator> alloc = nullptr, void *ptr = nullptr);

        explicit Tensor(DataType data_type, int32_t dim0, int32_t dim1, bool need_alloc = false,
                        std::shared_ptr<Allocator> alloc = nullptr, void *ptr = nullptr);

        explicit Tensor(DataType data_type, int32_t dim0, int32_t dim1, int32_t dim2,
                        bool need_alloc = false, std::shared_ptr<Allocator> alloc = nullptr,
                        void *ptr = nullptr);

        explicit Tensor(DataType data_type, int32_t dim0, int32_t dim1, int32_t dim2, int32_t dim3,
                        bool need_alloc = false, std::shared_ptr<Allocator> alloc = nullptr,
                        void *ptr = nullptr);

        explicit Tensor(DataType data_type, std::vector<int32_t> dims, bool need_alloc = false,
                        std::shared_ptr<Allocator> alloc = nullptr, void *ptr = nullptr);

        void to_cpu();

        void to_cuda(cudaStream_t stream = nullptr);

        bool is_empty() const;

        void init_buffer(std::shared_ptr<Allocator> alloc, DataType data_type,
                         bool need_alloc, void *ptr);

        template <typename T>
        T *ptr();

        template <typename T>
        const T *ptr() const;

        void reshape(const std::vector<int32_t> &dims);

        std::shared_ptr<Buffer> get_buffer() const;

        size_t size() const;

        size_t byte_size() const;

        int32_t dims_size() const;

        DataType data_type() const;

        int32_t get_dim(int32_t idx) const;

        const std::vector<int32_t> &dims() const;

        std::vector<size_t> strides() const;

        bool assign(std::shared_ptr<Buffer> buffer);

        void reset(DataType data_type, const std::vector<int32_t> &dims);

        void set_device_type(DeviceType device_type) const;

        DeviceType device_type() const;

        bool allocate(std::shared_ptr<Allocator> allocator, bool need_realloc = false);

        template <typename T>
        T *ptr(int64_t index);

        template <typename T>
        const T *ptr(int64_t index) const;

        template <typename T>
        T &index(int64_t offset);

        template <typename T>
        const T &index(int64_t offset) const;

        Tensor clone() const;

    private:
        size_t size_ = 0;
        std::vector<int32_t> dims_;
        std::shared_ptr<Buffer> buffer_;
        DataType data_type_ = DataType::kDataTypeUnknown;
    };
}