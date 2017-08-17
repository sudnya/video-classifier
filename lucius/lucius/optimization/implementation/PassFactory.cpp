/*  \file   PassFactory.cpp
    \author Gregory Diamos
    \date   May 4, 2017
    \brief  The header file for the PassFactory class.
*/

// Lucius Includes
#include <lucius/optimization/interface/PassFactory.h>

#include <lucius/optimization/interface/MemoryEfficientBackPropagationPass.h>
#include <lucius/optimization/interface/OperationDecomposerPass.h>

namespace lucius
{
namespace optimization
{

std::unique_ptr<Pass> PassFactory::create(const std::string& passName)
{
    if("MemoryEfficientBackPropagationPass" == passName)
    {
        return std::make_unique<MemoryEfficientBackPropagationPass>();
    }
    if("OperationDecomposerPass" == passName)
    {
        return std::make_unique<OperationDecomposerPass>();
    }

    return std::unique_ptr<Pass>();
}

} // namespace optimization
} // namespace lucius







