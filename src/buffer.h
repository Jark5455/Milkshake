//
// Created by yashr on 2/20/23.
//

#pragma once

class Buffer {
public:
    Buffer(size_t size);
    ~Buffer();

    void* getBufferHandle();
private:
    void* buffer;
};
