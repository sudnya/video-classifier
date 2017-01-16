
// Lucius Includes
#include <lucius/parallel/interface/Debug.h>

// Standard Library Includes
#include <iostream>

namespace lucius
{
namespace parallel
{

#if ENABLE_LOGGING
CUDA_MANAGED_DECORATOR LogDatabase* logDatabase = nullptr;
#endif

}
}


