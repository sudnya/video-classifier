/*  \file   MemoryAllocator.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the MemoryAllocator class.
*/

#pragma once

namespace lucius
{

namespace memory
{

class MemoryAllocatorImplementation;

/*! \brief An interface for a memory allocator component. */
class MemoryAllocator
{
public:
    MemoryAllocator();
    ~MemoryAllocator();

public:
    void* malloc(size_t size);
    void free(void* address);

private:
    std::unique_ptr<MemoryAllocatorImplementation> _implementation;

};

} // namespace memory
} // namespace lucius




