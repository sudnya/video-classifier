/*    \file   CudaRuntimeLibrary.cpp
    \date   Thursday August 15, 2013
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The source file for the CudaRuntimeLibrary class.
*/

// Lucious Includes
#include <lucious/parallel/interface/CudaRuntimeLibrary.h>

#include <lucious/parallel/interface/cuda.h>

#include <lucious/util/interface/Casts.h>
#include <lucious/util/interface/Knobs.h>

// Standard Library Includes
#include <stdexcept>

// System-Specific Includes
#include <dlfcn.h>

namespace lucious
{

namespace parallel
{

void CudaRuntimeLibrary::load()
{
    _interface.load();
}

bool CudaRuntimeLibrary::loaded()
{
	load();

    return _interface.loaded();

}

void CudaRuntimeLibrary::cudaSetDevice(int device)
{
    _check();

    int status = (*_interface.cudaSetDevice)(device);

    if(status != cudaSuccess)
    {
        throw std::runtime_error("Cuda set device failed: " +
            cudaGetErrorString(status));
    }
}

void CudaRuntimeLibrary::cudaDeviceSynchronize()
{
    _check();

    int status = (*_interface.cudaDeviceSynchronize)();

    if(status != cudaSuccess)
    {
        throw std::runtime_error("Cuda device synchronize failed: " +
            cudaGetErrorString(status));
    }
}

void* CudaRuntimeLibrary::cudaMalloc(size_t bytes)
{
    _check();

    void* address = nullptr;

    int status = (*_interface.cudaMalloc)(&address, bytes);

    if(status != cudaSuccess)
    {
        throw std::runtime_error("Cuda malloc failed: " +
            cudaGetErrorString(status));
    }

    //util::log("CudaRuntimeLibrary") << " CUDA allocated memory (address: "
    //    << address << ", " << bytes << " bytes)\n";

    return address;
}

void* CudaRuntimeLibrary::cudaMallocManaged(size_t bytes)
{
    _check();

    void* address = nullptr;

    int status = (*_interface.cudaMallocManaged)(&address, bytes, 1);

    if(status != cudaSuccess)
    {
        throw std::runtime_error("Cuda malloc managed failed: " +
            cudaGetErrorString(status));
    }

    //util::log("CudaRuntimeLibrary") << " CUDA allocated memory (address: "
    //    << address << ", " << bytes << " bytes)\n";

    return address;
}

void* CudaRuntimeLibrary::cudaHostAlloc(size_t bytes)
{
    _check();

    void* address = nullptr;

    int status = (*_interface.cudaHostAlloc)(&address, bytes, cudaHostAllocMapped);

    if(status != cudaSuccess)
    {
        throw std::runtime_error("Cuda host alloc failed: " +
            cudaGetErrorString(status));
    }

    //util::log("CudaRuntimeLibrary") << " CUDA allocated memory (address: "
    //    << address << ", " << bytes << " bytes)\n";

    return address;
}

void CudaRuntimeLibrary::cudaFree(void* ptr)
{
    _check();

    int status = (*_interface.cudaFree)(ptr);

    if(status != cudaSuccess)
    {
        throw std::runtime_error("Cuda free failed: " +
            cudaGetErrorString(status));
    }

    //util::log("CudaRuntimeLibrary") << " CUDA freed memory (address: "
    //    << ptr << ")\n";
}

void CudaRuntimeLibrary::cudaFreeHost(void* ptr)
{
    _check();

    int status = (*_interface.cudaFreeHost)(ptr);

    if(status != cudaSuccess)
    {
        throw std::runtime_error("Cuda free host failed: " +
            cudaGetErrorString(status));
    }

    //util::log("CudaRuntimeLibrary") << " CUDA freed memory (address: "
    //    << ptr << ")\n";
}

void CudaRuntimeLibrary::cudaMemcpy(void* dest, const void* src, size_t bytes,
    cudaMemcpyKind kind)
{
    _check();

    //util::log("CudaRuntimeLibrary") << " CUDA memcpy (destination address: "
    //    << dest << ", source address: " << src << ", " << bytes
    //    << " bytes)\n";

    int status = (*_interface.cudaMemcpy)(dest, src, bytes, kind);

    if(status != cudaSuccess)
    {
        throw std::runtime_error("Cuda memcpy failed: " +
            cudaGetErrorString(status));
    }
}

std::string CudaRuntimeLibrary::cudaGetErrorString(int error)
{
    _check();

    return (*_interface.cudaGetErrorString)(error);
}

void CudaRuntimeLibrary::_check()
{
    load();

    if(!loaded())
    {
        throw std::runtime_error("Tried to call CUDA runtime function when "
            "the library is not loaded. Loading library failed, consider "
            "installing the CUDA runtime.");
    }
}

CudaRuntimeLibrary::Interface::Interface()
: _library(nullptr), _failed(false)
{

}

CudaRuntimeLibrary::Interface::~Interface()
{
    unload();
}

static void checkFunction(void* pointer, const std::string& name)
{
    if(pointer == nullptr)
    {
        throw std::runtime_error("Failed to load function '" + name +
            "' from dynamic library.");
    }
}

void CudaRuntimeLibrary::Interface::load()
{
    if(_failed)  return;
    if(loaded()) return;
    if(!util::KnobDatabase::getKnobValue("Cuda::Enable", 1)) return;

    #ifdef __APPLE__
    const char* libraryName = "libcudart.dylib";
    #else
    const char* libraryName = "libcudart.so";
    #endif

    _library = dlopen(libraryName, RTLD_LAZY);

    util::log("CudaRuntimeLibrary") << "Loading library '" << libraryName << "'\n";

    if(!loaded())
    {
        util::log("CudaRuntimeLibrary") << " Failed to load library '" << libraryName
            << "'\n";
        _failed = true;
        return;
    }

    try
    {

        #define DynLink( function ) \
            util::bit_cast(function, dlsym(_library, #function)); \
            checkFunction((void*)function, #function)

        DynLink(cudaSetDevice);
        DynLink(cudaDeviceSynchronize);
        DynLink(cudaMalloc);
        DynLink(cudaMallocManaged);
        DynLink(cudaHostAlloc);
        DynLink(cudaFree);
        DynLink(cudaFreeHost);
        DynLink(cudaMemcpy);
        DynLink(cudaGetErrorString);

        #undef DynLink

        CudaRuntimeLibrary::cudaSetDevice(0);

        util::log("CudaRuntimeLibrary") << " Loaded library '" << libraryName
            << "' successfully\n";
    }
    catch(...)
    {
        unload();
        throw;
    }
}

bool CudaRuntimeLibrary::Interface::loaded() const
{
    return _library != nullptr;
}

void CudaRuntimeLibrary::Interface::unload()
{
    if(!loaded()) return;

    dlclose(_library);
    _library = nullptr;
}

CudaRuntimeLibrary::Interface CudaRuntimeLibrary::_interface;

}

}



