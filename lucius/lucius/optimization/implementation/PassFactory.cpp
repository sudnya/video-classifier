/*  \file   PassFactory.cpp
    \author Gregory Diamos
    \date   May 4, 2017
    \brief  The header file for the PassFactory class.
*/

// Lucius Includes
#include <lucius/optimization/interface/PassFactory.h>

#include <lucius/optimization/interface/MemoryEfficientBackPropagationPass.h>
#include <lucius/optimization/interface/OperationDecomposerPass.h>
#include <lucius/optimization/interface/TableOperationSelectionPass.h>
#include <lucius/optimization/interface/MinimalMemoryOperationSchedulingPass.h>
#include <lucius/optimization/interface/DynamicMemoryAllocationPass.h>
#include <lucius/optimization/interface/OperationFinalizationPass.h>
#include <lucius/optimization/interface/LazyProgramCompleterPass.h>

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
    if("TableOperationSelectionPass" == passName)
    {
        return std::make_unique<TableOperationSelectionPass>();
    }
    if("MinimalMemoryOperationSchedulingPass" == passName)
    {
        return std::make_unique<MinimalMemoryOperationSchedulingPass>();
    }
    if("DynamicMemoryAllocationPass" == passName)
    {
        return std::make_unique<DynamicMemoryAllocationPass>();
    }
    if("OperationFinalizationPass" == passName)
    {
        return std::make_unique<OperationFinalizationPass>();
    }
    if("LazyProgramCompleterPass" == passName)
    {
        return std::make_unique<LazyProgramCompleterPass>();
    }

    return std::unique_ptr<Pass>();
}

} // namespace optimization
} // namespace lucius







