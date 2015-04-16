

#include <minerva/parallel/interface/cuda.h>

namespace minerva
{

namespace parallel
{

bool isCudaEnabled()
{
	#ifdef __NVCC__
	return true;
	#else
	return false;
	#endif
}

}
}

