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

#define CHECK_EQ(x, y) assert((x) == (y))


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