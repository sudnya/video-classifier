
#pragma once

// Standard Library Includes
#include <functional>

// Forward Declarations
namespace minerva { namespace parallel { class ThreadGroup; } }

namespace minerva
{
namespace parallel
{

void parallelFor(const std::function<void(parallel::ThreadGroup g)>& function);

}
}





