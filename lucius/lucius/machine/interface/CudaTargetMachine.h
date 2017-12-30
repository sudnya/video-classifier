/*  \file   CudaTargetMachine.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the CudaTargetMachineclass.
*/

#pragma once

// Lucius Includes
#include <lucius/machine/interface/TargetMachineInterface.h>

namespace lucius
{

namespace machine
{

/*! \brief A class for representing a Cuda target for code generation. */
class CudaTargetMachine : public TargetMachineInterface
{
public:
    virtual ~CudaTargetMachine();

public:
    virtual TableEntryVector getEntries() const;

public:
    virtual std::unique_ptr<ir::TargetOperationFactory> getOperationFactory() const;

public:
    virtual std::string name() const;

};

}

}









