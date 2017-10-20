/*  \file   OperationMemoryAnalysis.cpp
    \author Gregory Diamos
    \date   May 21, 2017
    \brief  The source file for the OperationMemoryAnalysis class.
*/

// Lucius Includes
#include <lucius/analysis/interface/OperationMemoryAnalysis.h>

#include <lucius/ir/interface/Type.h>
#include <lucius/ir/interface/Value.h>
#include <lucius/ir/interface/Operation.h>

// Standard Library Incldues
#include <cassert>

namespace lucius
{

namespace analysis
{

OperationMemoryAnalysis::OperationMemoryAnalysis()
{

}

OperationMemoryAnalysis::~OperationMemoryAnalysis()
{

}

static bool isTypeMemoryUsageStatic(const ir::Type& type)
{
    if(type.isScalar())
    {
        return true;
    }

    // TODO: handle more interesting cases

    return false;
}

static double getMemoryUsage(const ir::Value& value)
{
    assert(isTypeMemoryUsageStatic(value.getType()));

    // TODO: handle cases where the memory usage is not static

    return value.getType().getBytes();
}

double OperationMemoryAnalysis::getOperationMemoryRequirement(const ir::Operation& operation) const
{
    double memoryUsage = 0.0;

    auto values = operation.getUsedValues();

    for(auto& value : values)
    {
        memoryUsage += getMemoryUsage(value);
    }

    // TODO: handle operations with scratch requirements

    return memoryUsage;
}

double OperationMemoryAnalysis::getOperationSavedMemoryRequirement(
    const ir::Operation& operation) const
{
    return getOperationMemoryRequirement(operation);
}

void OperationMemoryAnalysis::runOnFunction(const ir::Function& function)
{
    // TODO
}

OperationMemoryAnalysis::StringSet OperationMemoryAnalysis::getRequiredAnalyses() const
{
    return StringSet({});
}

} // namespace analysis
} // namespace lucius



