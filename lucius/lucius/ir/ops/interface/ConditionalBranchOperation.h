/*  \file   ConditionalBranchOperation.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the ConditionalBranchOperation class.
*/

#pragma once

// Lucius Includes
#include <lucius/ir/ops/interface/ControlOperation.h>

namespace lucius
{

namespace ir
{

/*! \brief A class for representing a conditional branch. */
class ConditionalBranchOperation : public ControlOperation
{
public:
    ConditionalBranchOperation(Value predicate, BasicBlock target, BasicBlock fallthrough);
    virtual ~ConditionalBranchOperation();

};

} // namespace ir
} // namespace lucius


