
// Lucius Includes
#include <lucius/parallel/interface/Synchronization.h>
#include <lucius/parallel/interface/CudaRuntimeLibrary.h>

// Standard Library Includes
#include <atomic>

namespace lucius
{

namespace parallel
{

static std::atomic<bool> isSynchronized(true);

void setNotSynchronized()
{
    isSynchronized.store(false, std::memory_order_release);
}

void synchronize()
{
    if(!CudaRuntimeLibrary::loaded())
    {
        return;
    }

    if(isSynchronized.load(std::memory_order_acquire))
    {
        return;
    }

    CudaRuntimeLibrary::cudaDeviceSynchronize();

    isSynchronized.store(true, std::memory_order_release);
}

}

}





