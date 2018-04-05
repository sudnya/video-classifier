/*  \file   IRExecutionEngineOptions.cpp
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The source file for the IRExecutionEngineOptions class.
*/

// Lucius Includes
#include <lucius/runtime/interface/IRExecutionEngineOptions.h>

namespace lucius
{

namespace runtime
{

void IRExecutionEngineOptions::addTargetIndependentOptimizationPass(const std::string& passName)
{
    _targetIndependentOptimizationPasses.push_back(passName);
}

const IRExecutionEngineOptions::StringList&
IRExecutionEngineOptions::getTargetIndependentOptimizationPasses() const
{
    return _targetIndependentOptimizationPasses;
}


} // namespace runtime
} // namespace lucius




