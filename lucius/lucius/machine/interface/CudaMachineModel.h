/*  \file   CudaMachineModel.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the CudaMachineModel class.
*/

#pragma once

// Lucius Include
#include <lucius/machine/interface/MachineModelInterface.h>

namespace lucius
{

namespace machine
{

/*! \brief A class for representing the machine that is executing a lucius program. */
class CudaMachineModel : public MachineModelInterface
{
public:
    /*! \brief Get the launch overhead of an operation in seconds. */
    double getOperationLaunchOverhead() const;

    /*! \brief Get the floating point throughput of the machine in ops/second. */
    double getFloatingPointThroughput() const;

    /*! \brief Get the memory throughout of the machine in bytes/second. */
    double getMemoryOperationThroughput() const;

    /*! \brief Get the network throughput of the machine in bytes/second. */
    double getNetworkOperationThroughput() const;

public:
    /*! \brief Get the total memory capacity of the system. */
    double getTotalSystemMemoryCapacity() const;
};

}

}








