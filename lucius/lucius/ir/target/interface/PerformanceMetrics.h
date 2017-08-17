/*  \file   PerformanceMetrics.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the PerformanceMetrics class.
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

public:
    /*! \brief Get the total number of flops performed by the operation. */
    double getTotalFloatingPointOperations() const;

    /*! \brief Get the total number of bytes transferred to memory by the operation. */
    double getTotalMemoryOperations() const;

    /*! \brief Get the total number of bytes transferred to network by the operation. */
    double getTotalNetworkOperations() const;

private:
    double _floatingPointOperations;
    double _memoryOperations;
    double _networkOperations;

};

} // namespace ir
} // namespace lucius




