/*    \file   CudaRuntimeLibrary.h
    \date   Thursday August 15, 2013
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The header file for the CudaRuntimeLibrary class.
*/

#pragma once

// Standard Library Includes
#include <cstring>
#include <string>

namespace lucious
{

namespace parallel
{

class CudaRuntimeLibrary
{
public:
    enum cudaMemcpyKind
    {
        cudaMemcpyHostToHost          =   0,
        cudaMemcpyHostToDevice        =   1,
        cudaMemcpyDeviceToHost        =   2,
        cudaMemcpyDeviceToDevice      =   3,
        cudaMemcpyDefault             =   4
    };

    enum cudaResult
    {
        cudaSuccess = 0
    };

    enum cudaHostAllocFlag
    {
        cudaHostAllocMapped = 0x2
    };

public:
    static void load();
    static bool loaded();

public:
    static void cudaSetDevice(int device);
    static void cudaDeviceSynchronize();

    static void* cudaMalloc(size_t bytes);
    static void* cudaMallocManaged(size_t bytes);
    static void* cudaHostAlloc(size_t bytes);
    static void cudaFree(void* ptr);
    static void cudaFreeHost(void* ptr);

    static void cudaMemcpy(void* dest, const void* src,
        size_t bytes, cudaMemcpyKind kind = cudaMemcpyDefault);

public:
    static std::string cudaGetErrorString(int error);

private:
    static void _check();

private:
    class Interface
    {
    public:
        int (*cudaSetDevice)(int ptr);
        int (*cudaDeviceSynchronize)();

        int (*cudaMalloc)(void** ptr, size_t bytes);
        int (*cudaMallocManaged)(void** ptr, size_t bytes, unsigned int flags);
        int (*cudaHostAlloc)(void** ptr, size_t bytes, unsigned int flags);
        int (*cudaFree)  (void*  ptr);
        int (*cudaFreeHost)  (void*  ptr);
        int (*cudaMemcpy)(void*  dest, const void* src, size_t bytes,
            cudaMemcpyKind kind);

    public:
        const char* (*cudaGetErrorString)(int error);

    public:
        /*! \brief The constructor zeros out all of the pointers */
        Interface();

        /*! \brief The destructor closes dlls */
        ~Interface();
        /*! \brief Load the library */
        void load();
        /*! \brief Has the library been loaded? */
        bool loaded() const;
        /*! \brief unloads the library */
        void unload();

    private:
        void* _library;
        bool  _failed;

    };

private:
    static Interface _interface;

};

}

}

