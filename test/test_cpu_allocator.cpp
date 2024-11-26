#include "cpu_allocator.h"
#include <gtest/gtest.h>

TEST(CPUAllocatorTest, AllocateZeroBytes) {
    CPU_Allocator allocator;
    void* ptr = allocator.allocate(0);
    EXPECT_EQ(ptr, nullptr);
}

TEST(CPUAllocatorTest, AllocateNonZeroBytes) {
    CPU_Allocator allocator;
    size_t size = 1024;
    void* ptr = allocator.allocate(size);
    EXPECT_NE(ptr, nullptr);
    allocator.release(ptr);
}

TEST(CPUAllocatorTest, ReleaseNullptr) {
    CPU_Allocator allocator;
    allocator.release(nullptr);
    SUCCEED();
}

TEST(CPUAllocatorTest, ReleaseValidPtr) {
    CPU_Allocator allocator;
    size_t size = 1024;
    void* ptr = allocator.allocate(size);
    EXPECT_NE(ptr, nullptr);
    allocator.release(ptr);
}

TEST(CPUAllocatorTest, AllocateAndReleaseMultipleTimes) {
    CPU_Allocator allocator;
    for (int i = 0; i < 10; ++i) {
        size_t size = 1024;
        void* ptr = allocator.allocate(size);
        EXPECT_NE(ptr, nullptr);
        allocator.release(ptr);
    }
}

TEST(CPUAllocatorTest, AllocateLargeSize) {
    CPU_Allocator allocator;
    size_t size = 1024 * 1024 * 1024; // 1 GB
    void* ptr = allocator.allocate(size);
    EXPECT_NE(ptr, nullptr);
    allocator.release(ptr);
}