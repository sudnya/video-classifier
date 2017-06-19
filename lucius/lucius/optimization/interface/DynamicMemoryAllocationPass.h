/*  \file   DynamicMemoryAllocationPass.h
    \author Gregory Diamos
    \date   May 4, 2017
    \brief  The header file for the DynamicMemoryAllocatin class.
*/

#pragma once

// Lucius Includes
#include <lucius/optimization/interface/Pass.h>

namespace lucius
{
namespace optimization
{

/*! \brief A class that schdules operations to minimize memory consumption. */
class DynamicMemoryAllocationPass : public Pass
{
public:
    DynamicMemoryAllocationPass();
    virtual ~DynamicMemoryAllocationPass();

public:
    void runOnFunction(ir::Function& ) final;

public:
    StringSet getRequiredAnalyses() const final;
};

} // namespace optimization
} // namespace lucius






