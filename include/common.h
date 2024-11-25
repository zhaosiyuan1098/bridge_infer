#ifndef COMMON_H
#define COMMON_H
#include"stddef.h"
#include"stdint-gcc.h"
#include<map>
#include<vector>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>
#include <cassert>
#include <memory>
#include <cuda_runtime_api.h>
#include<cstring>

#define CHECK_EQ(x, y) assert((x) == (y))
#define CHECK_NE(x, y) assert((x) != (y))
#define CHECK(cond) \
    do { \
        if (!(cond)) { \
            std::cerr << "Check failed: " << #cond << " in file " << __FILE__ << " at line " << __LINE__ << std::endl; \
            std::abort(); \
        } \
    } while (0)

enum class DeviceType : uint8_t
{
  kDeviceUnknown = 0,
  kDeviceCPU = 1,
  kDeviceCUDA = 2,
};
enum class MemcpyKind
{
  kMemcpyCPU2CPU = 0,
  kMemcpyCPU2CUDA = 1,
  kMemcpyCUDA2CPU = 2,
  kMemcpyCUDA2CUDA = 3,
};


#endif