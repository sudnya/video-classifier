/*  \file   OperationMemoryAnalysis.cpp
    \author Gregory Diamos
    \date   May 21, 2017
    \brief  The source file for the OperationMemoryAnalysis class.
*/

// Lucius Includes
#include <lucius/analysis/interface/OperationMemoryAnalysis.h>

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

static double getMemoryUsage(const ir::Value* value)
{
    return value->getBytes();
}

double OperationMemoryAnalysis::getOperationMemoryRequirement(const ir::Operation* operation) const
{
    double memoryUsage = 0.0;

    auto values = operation->getAllValues();

    for(auto* value : values)
    {
        memoryUsage += getMemoryUsage(value);
    }

    // TODO: handle operations with scratch requirements

    return memoryUsage;
}

void OperationMemoryAnalysis::runOnFunction(const ir::Function& function)
{

}

StringSet OperationMemoryAnalysis::getRequiredAnalyses() const
{
    return StringSet({});
}

} // namespace analysis
} // namespace lucius



