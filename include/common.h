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
#define CHECK(condition)                             \
    if (!(condition))                                \
    {                                                \
        std::cerr << "Check failed: " << #condition; \
        std::abort();                                \
    }

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
enum class LayerType : uint8_t {
  kLayerUnknown = 0,
  kLayerLinear = 1,
  kLayerEncode = 2,
  kLayerEmbedding = 3,
  kLayerRMSNorm = 4,
  kLayerMatmul = 5,
  kLayerRoPe = 6,
  kLayerMHA = 7,
  kLayerSoftmax = 8,
  kLayerAdd = 9,
  kLayerSwiGLU = 10,
};

enum class DataType : uint8_t {
  kDataTypeUnknown = 0,
  kDataTypeFp32 = 1,
  kDataTypeInt8 = 2,
  kDataTypeInt32 = 3,
};

enum StatusCode : uint8_t {
  kSuccess = 0,
  kFunctionUnImplement = 1,
  kPathNotValid = 2,
  kModelParseError = 3,
  kInternalError = 5,
  kKeyValueHasExist = 6,
  kInvalidArgument = 7,
};

class Status {
 public:
  Status(int code = StatusCode::kSuccess, std::string err_message = "");

  Status(const Status& other) = default;

  Status& operator=(const Status& other) = default;

  Status& operator=(int code);

  bool operator==(int code) const;

  bool operator!=(int code) const;

  operator int() const;

  operator bool() const;

  int32_t get_err_code() const;

  const std::string& get_err_msg() const;

  void set_err_msg(const std::string& err_msg);

 private:
  int code_ = StatusCode::kSuccess;
  std::string message_;
};


#endif