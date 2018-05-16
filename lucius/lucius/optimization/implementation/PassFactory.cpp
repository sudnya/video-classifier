/*  \file   PassFactory.cpp
    \author Gregory Diamos
    \date   May 4, 2017
    \brief  The header file for the PassFactory class.
*/

// Lucius Includes
#include <lucius/optimization/interface/PassFactory.h>

#include <lucius/optimization/interface/ConvertLazyProgramToSSAPass.h>
#include <lucius/optimization/interface/MemoryEfficientBackPropagationPass.h>
#include <lucius/optimization/interface/OperationDecomposerPass.h>
#include <lucius/optimization/interface/TableOperationSelectionPass.h>
#include <lucius/optimization/interface/MinimalMemoryOperationSchedulingPass.h>
#include <lucius/optimization/interface/DynamicMemoryAllocationPass.h>
#include <lucius/optimization/interface/OperationFinalizationPass.h>
#include <lucius/optimization/interface/LazyProgramCompleterPass.h>
#include <lucius/optimization/interface/LowerVariablesPass.h>

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
    else if("OperationDecomposerPass" == passName)
    {
        return std::make_unique<OperationDecomposerPass>();
    }
    else if("TableOperationSelectionPass" == passName)
    {
        return std::make_unique<TableOperationSelectionPass>();
    }
    else if("MinimalMemoryOperationSchedulingPass" == passName)
    {
        return std::make_unique<MinimalMemoryOperationSchedulingPass>();
    }
    else if("DynamicMemoryAllocationPass" == passName)
    {
        return std::make_unique<DynamicMemoryAllocationPass>();
    }
    else if("OperationFinalizationPass" == passName)
    {
        return std::make_unique<OperationFinalizationPass>();
    }
    else if("LazyProgramCompleterPass" == passName)
    {
        return std::make_unique<LazyProgramCompleterPass>();
    }
    else if("LowerVariablesPass" == passName)
    {
        return std::make_unique<LowerVariablesPass>();
    }

    return std::unique_ptr<Pass>();
}

std::unique_ptr<Pass> PassFactory::create(const std::string& passName, const util::Any& parameters)
{
    if("ConvertLazyProgramToSSAPass" == passName)
    {
        return std::make_unique<ConvertLazyProgramToSSAPass>(parameters);
    }

    return std::unique_ptr<Pass>();
}

} // namespace optimization
} // namespace lucius







