/*  \file   PerformanceMetrics.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the Operation class.
*/

#pragma once

namespace lucius
{

namespace ir
{

/*! \brief Performance metrics for an operation. */
class PerformanceMetrics
{
public:
    PerformanceMetrics(double flops, double mops, double nops);

private:
    double _floatingPointOperations;
    double _memoryOperations;
    double _networkOperations;

};

} // namespace ir
} // namespace lucius




