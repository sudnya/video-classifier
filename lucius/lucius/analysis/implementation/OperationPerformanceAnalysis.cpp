/*  \file   OperationPerformanceAnalysis.cpp
    \author Gregory Diamos
    \date   May 21, 2017
    \brief  The source file for the OperationPerformanceAnalysis class.
*/

// Lucius Includes
#include <lucius/analysis/interface/OperationPerformanceAnalysis.h>

namespace lucius
{

namespace analysis
{

OperationPerformanceAnalysis::OperationPerformanceAnalysis()
{

}

OperationPerformanceAnalysis::~OperationPerformanceAnalysis()
{
    // intentionally blank
}

double OperationPerformanceAnalysis::getOperationTime(const ir::Operation* operation) const
{
    PerformanceMetrics metrics = operation->getPerformanceMetrics();

    // roofline model
    double computeTime = metrics.getTotalFloatingPointOperations() /
                         machine::MachineModel::getFloatingPointThroughput();


    double memoryTime = metrics.getTotalMemoryOperations() /
                        machine::MachineModel::getMemoryOperationThroughput();

    double networkTime = metrics.getTotalNetworkOperations() /
                         machine::MachineModel::getNetworkOperationThroughput();


    return std::max(computeTime, memoryTime, networkTime);
}

double OperationPerformanceAnalysis::getOverheadTime(const ir::Operation* operation) const
{
    return machine::MachineModel::getOperationLaunchOverhead();
}

void OperationPerformanceAnalysis::runOnFunction(const ir::Function& function)
{
    // intentionally empty
}

StringSet OperationPerformanceAnalysis::getRequiredAnalyses() const
{
    return StringSet({});
}

} // namespace analysis
} // namespace lucius



