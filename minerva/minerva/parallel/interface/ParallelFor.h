
#pragma once

// Standard Library Includes
#include <functional>

// Forward Declarations
namespace lucious { namespace parallel { class ThreadGroup; } }

namespace lucious
{
namespace parallel
{

void parallelFor(const std::function<void(parallel::ThreadGroup g)>& function);

}
}





