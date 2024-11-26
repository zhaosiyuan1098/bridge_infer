#include "cuda_allocator.h"
#include <gtest/gtest.h>

class CUDAAllocatorTest : public ::testing::Test {
protected:
    CUDA_Allocator allocator;

    void SetUp() override {
        // Any setup code if needed
    }

    void TearDown() override {
        // Any cleanup code if needed
    }
};

TEST_F(CUDAAllocatorTest, AllocateZeroBytes) {
    void* ptr = allocator.allocate(0);
    EXPECT_EQ(ptr, nullptr);
}

TEST_F(CUDAAllocatorTest, AllocateNonZeroBytes) {
    size_t byte_size = 1024;
    void* ptr = allocator.allocate(byte_size);
    EXPECT_NE(ptr, nullptr);
    allocator.release(ptr);
}

TEST_F(CUDAAllocatorTest, ReleaseNullPointer) {
    void* ptr = nullptr;
    allocator.release(ptr);
    // No assertion needed, just ensuring no crash
}

TEST_F(CUDAAllocatorTest, AllocateAndRelease) {
    size_t byte_size = 2048;
    void* ptr = allocator.allocate(byte_size);
    EXPECT_NE(ptr, nullptr);
    allocator.release(ptr);
    // Ensure the pointer is released without error
}

