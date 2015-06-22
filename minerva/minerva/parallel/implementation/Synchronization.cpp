
// Minerva Includes
#include <minerva/parallel/interface/Synchronization.h>
#include <minerva/parallel/interface/CudaRuntimeLibrary.h>

// Standard Library Includes
#include <atomic>

namespace minerva
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





