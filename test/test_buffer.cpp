#include "buffer.h"
#include "cpu_allocator.h"
#include "cuda_allocator.h"
#include <gtest/gtest.h>
#include <cstring>

TEST(BufferTest, CopyFromCPUToCPU) {
    size_t size = 1024;
    auto allocator = std::make_shared<CPU_Allocator>();

    // Create source buffer with some data
    Buffer srcBuffer(size, allocator);
    int* srcData = static_cast<int*>(srcBuffer.ptr());
    for (size_t i = 0; i < size / sizeof(int); ++i) {
        srcData[i] = static_cast<int>(i);
    }

    // Create destination buffer
    Buffer destBuffer(size, allocator);

    // Copy data
    destBuffer.copy_from(srcBuffer);

    // Verify data
    int* destData = static_cast<int*>(destBuffer.ptr());
    EXPECT_EQ(0, memcmp(srcData, destData, size));
}

TEST(BufferTest, CopyFromCPUToCPU_DifferentSizes) {
    size_t srcSize = 2048;
    size_t destSize = 1024;
    auto allocator = std::make_shared<CPU_Allocator>();

    // Create source buffer with some data
    Buffer srcBuffer(srcSize, allocator);
    int* srcData = static_cast<int*>(srcBuffer.ptr());
    for (size_t i = 0; i < srcSize / sizeof(int); ++i) {
        srcData[i] = static_cast<int>(i);
    }

    // Create destination buffer with smaller size
    Buffer destBuffer(destSize, allocator);

    // Copy data
    destBuffer.copy_from(srcBuffer);

    // Verify data (only up to destSize)
    int* destData = static_cast<int*>(destBuffer.ptr());
    EXPECT_EQ(0, memcmp(srcData, destData, destSize));
}

#ifdef USE_CUDA
TEST(BufferTest, CopyFromCPUToCUDA) {
    size_t size = 1024;
    auto cpuAllocator = std::make_shared<CPU_Allocator>();
    auto cudaAllocator = std::make_shared<CUDA_Allocator>();

    // Create source buffer on CPU with some data
    Buffer srcBuffer(size, cpuAllocator);
    int* srcData = static_cast<int*>(srcBuffer.ptr());
    for (size_t i = 0; i < size / sizeof(int); ++i) {
        srcData[i] = static_cast<int>(i);
    }

    // Create destination buffer on CUDA device
    Buffer destBuffer(size, cudaAllocator);

    // Copy data from CPU to CUDA buffer
    destBuffer.copy_from(srcBuffer);

    // Copy data back to CPU for verification
    Buffer verifyBuffer(size, cpuAllocator);
    verifyBuffer.copy_from(destBuffer);

    int* verifyData = static_cast<int*>(verifyBuffer.ptr());
    EXPECT_EQ(0, memcmp(srcData, verifyData, size));
}

TEST(BufferTest, CopyFromCUDAToCPU) {
    size_t size = 1024;
    auto cpuAllocator = std::make_shared<CPU_Allocator>();
    auto cudaAllocator = std::make_shared<CUDA_Allocator>();

    // Create source buffer on CUDA device with some data
    Buffer srcBuffer(size, cudaAllocator);

    // Initialize data on host
    int* hostData = new int[size / sizeof(int)];
    for (size_t i = 0; i < size / sizeof(int); ++i) {
        hostData[i] = static_cast<int>(i);
    }

    // Copy data to CUDA buffer
    cudaMemcpy(srcBuffer.ptr(), hostData, size, cudaMemcpyHostToDevice);

    // Create destination buffer on CPU
    Buffer destBuffer(size, cpuAllocator);

    // Copy data from CUDA to CPU buffer
    destBuffer.copy_from(srcBuffer);

    // Verify data
    int* destData = static_cast<int*>(destBuffer.ptr());
    EXPECT_EQ(0, memcmp(hostData, destData, size));

    delete[] hostData;
}

TEST(BufferTest, CopyFromCUDAToCUDA) {
    size_t size = 1024;
    auto cudaAllocator = std::make_shared<CUDA_Allocator>();

    // Create source buffer on CUDA device with some data
    Buffer srcBuffer(size, cudaAllocator);

    // Initialize data on host
    int* hostData = new int[size / sizeof(int)];
    for (size_t i = 0; i < size / sizeof(int); ++i) {
        hostData[i] = static_cast<int>(i);
    }

    // Copy data to source CUDA buffer
    cudaMemcpy(srcBuffer.ptr(), hostData, size, cudaMemcpyHostToDevice);

    // Create destination buffer on CUDA device
    Buffer destBuffer(size, cudaAllocator);

    // Copy data from source to destination CUDA buffer
    destBuffer.copy_from(srcBuffer);

    // Copy data back to host for verification
    int* verifyData = new int[size / sizeof(int)];
    cudaMemcpy(verifyData, destBuffer.ptr(), size, cudaMemcpyDeviceToHost);

    // Verify data
    EXPECT_EQ(0, memcmp(hostData, verifyData, size));

    delete[] hostData;
    delete[] verifyData;
}
#endif

TEST(BufferTest, CopyFromNullBuffer) {
    size_t size = 1024;
    auto allocator = std::make_shared<CPU_Allocator>();

    // 创建目标缓冲区
    Buffer destBuffer(size, allocator);

    // 尝试从空缓冲区指针复制
    const Buffer* nullBuffer = nullptr;
    EXPECT_DEATH(destBuffer.copy_from(nullBuffer), ".*Check failed:.*");
}

TEST(BufferTest, CopyFromBufferWithNullPtr) {
    size_t size = 1024;

    // 创建未分配内存的源缓冲区（allocator 为 nullptr）
    Buffer srcBuffer(size, nullptr, nullptr);

    // 创建目标缓冲区
    auto allocator = std::make_shared<CPU_Allocator>();
    Buffer destBuffer(size, allocator);

    // 尝试从内部指针为 nullptr 的缓冲区复制
    EXPECT_DEATH(destBuffer.copy_from(srcBuffer), ".*Check failed:.*");
}