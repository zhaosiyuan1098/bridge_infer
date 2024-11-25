#include <iostream>
#include"common.h"
#include"allocator.h"
#include"buffer.h"

int main(int argc, char** argv) {

    std::shared_ptr<Allocator> allocator = std::make_shared<CPU_Allocator>();
    std::shared_ptr<Allocator> cuda_allocator = std::make_shared<CUDA_Allocator>();
    Buffer buffer(10, allocator);
    Buffer cuda_buffer(10, cuda_allocator);
    buffer.allocate();
    cuda_buffer.allocate();
    buffer.copy_from(cuda_buffer);
    cuda_buffer.copy_from(buffer);
    std::cout << "Hello, World!" << std::endl;
    return 0;
}
