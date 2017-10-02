/*  \file   PerformanceMetrics.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The source file for the PerformanceMetrics class.
*/

// Lucius Includes
#include <lucius/ir/target/interface/PerformanceMetrics.h>

namespace lucius
{

namespace ir
{

PerformanceMetrics::PerformanceMetrics(double flops, double mops, double nops)
: _floatingPointOperations(flops), _memoryOperations(mops), _networkOperations(nops)
{

}

double PerformanceMetrics::getTotalFloatingPointOperations() const
{
    return _floatingPointOperations;
}

double PerformanceMetrics::getTotalMemoryOperations() const
{
    return _memoryOperations;
}

double PerformanceMetrics::getTotalNetworkOperations() const
{
    return _networkOperations;
}

} // namespace ir
} // namespace lucius





