
// Lucius Includes
#include <lucius/parallel/interface/ParallelFor.h>
#include <lucius/parallel/interface/ConcurrentCollectives.h>
#include <lucius/parallel/interface/Synchronization.h>

// Standard Library Includes
#include <thread>
#include <list>

namespace lucius
{
namespace parallel
{

void parallelFor(const std::function<void(parallel::ThreadGroup g)>& function)
{
    typedef std::list<std::thread> ThreadList;

    size_t threadCount = std::thread::hardware_concurrency();

    ThreadList threads;

    for(size_t i = 0; i < threadCount; ++i)
    {
        threads.emplace_back(std::thread(function, ThreadGroup(threadCount, i)));
    }

    // barrier threads
    for(auto& thread : threads)
    {
        thread.join();
    }

    synchronize();
}

}
}






