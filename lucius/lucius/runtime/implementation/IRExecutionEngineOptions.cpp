/*  \file   IRExecutionEngineOptions.cpp
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The source file for the IRExecutionEngineOptions class.
*/

// Lucius Includes
#include <lucius/runtime/interface/IRExecutionEngineOptions.h>

#include <lucius/util/interface/Any.h>

namespace lucius
{

namespace runtime
{

void IRExecutionEngineOptions::addTargetIndependentOptimizationPass(const std::string& passName)
{
    _targetIndependentOptimizationPasses.push_back(PassDescriptor(passName, util::Any()));
}

void IRExecutionEngineOptions::addTargetIndependentOptimizationPass(const std::string& passName,
    const util::Any& anyValue)
{
    _targetIndependentOptimizationPasses.push_back(PassDescriptor(passName, anyValue));
}

const IRExecutionEngineOptions::PassList&
IRExecutionEngineOptions::getTargetIndependentOptimizationPasses() const
{
    return _targetIndependentOptimizationPasses;
}

void IRExecutionEngineOptions::addTargetDependentOptimizationPass(const std::string& passName)
{
    _targetDependentOptimizationPasses.push_back(PassDescriptor(passName, util::Any()));
}

const IRExecutionEngineOptions::PassList&
IRExecutionEngineOptions::getTargetDependentOptimizationPasses() const
{
    return _targetDependentOptimizationPasses;
}


} // namespace runtime
} // namespace lucius




