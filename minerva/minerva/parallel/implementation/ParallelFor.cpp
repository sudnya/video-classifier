
// Minerva Includes
#include <minerva/parallel/interface/ParallelFor.h>
#include <minerva/parallel/interface/ConcurrentCollectives.h>

// Standard Library Includes
#include <thread>
#include <list>

namespace minerva
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
	
	// synchronize cuda
	//CudaRuntime::cudaSynchronizeDevice(CudaRuntime::cudaGetCurrentDevice());
}

}
}





