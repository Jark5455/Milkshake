#include <stdexcept>

#include "buffer.h"

Buffer::Buffer(size_t size) {
    if (cudaMalloc(&buffer, size) != cudaSuccess){
        throw std::runtime_error("Failed to allocate buffer");
    }
}

Buffer::~Buffer() {
    if (cudaFree(buffer) != cudaSuccess) {
        throw std::runtime_error("Failed to destroy buffer");
    }
}

void* Buffer::getBufferHandle() {
    return  buffer;
}
