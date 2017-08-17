/*  \file   MachineModel.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the MachineModel class.
*/

#pragma once


namespace lucius
{

namespace machine
{

/*! \brief A class for representing the machine that is executing a lucius program. */
class MachineModel
{
public:
    /*! \brief Get the launch overhead of an operation in seconds. */
    static double getOperationLaunchOverhead();

    /*! \brief Get the floating point throughput of the machine in ops/second. */
    static double getFloatingPointThroughput();

    /*! \brief Get the memory throughout of the machine in bytes/second. */
    static double getMemoryOperationThroughput();

    /*! \brief Get the network throughput of the machine in bytes/second. */
    static double getNetworkOperationThroughput();

public:
    /*! \brief Get the total memory capacity of the system. */
    static double getTotalSystemMemoryCapacity();
};

}

}






