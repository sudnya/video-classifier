/*  \file   MachineModelInterface.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the MachineModelInterface class.
*/

#pragma once

namespace lucius
{

namespace machine
{

/*! \brief A class for representing the machine that is executing a lucius program. */
class MachineModelInterface
{
public:
    virtual ~MachineModelInterface();

public:
    /*! \brief Get the launch overhead of an operation in seconds. */
    virtual double getOperationLaunchOverhead() const = 0;

    /*! \brief Get the floating point throughput of the machine in ops/second. */
    virtual double getFloatingPointThroughput() const = 0;

    /*! \brief Get the memory throughout of the machine in bytes/second. */
    virtual double getMemoryOperationThroughput() const = 0;

    /*! \brief Get the network throughput of the machine in bytes/second. */
    virtual double getNetworkOperationThroughput() const = 0;

public:
    /*! \brief Get the total memory capacity of the system. */
    virtual double getTotalSystemMemoryCapacity() const = 0;

};

}

}







