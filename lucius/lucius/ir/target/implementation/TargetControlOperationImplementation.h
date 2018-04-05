/*  \file   TargetControlOperationImplementation.h
    \author Gregory Diamos
    \date   October 4, 2017
    \brief  The header file for the TargetControlOperationImplementation class.
*/

#pragma once

// Lucius Includes
#include <lucius/ir/target/implementation/TargetOperationImplementation.h>

namespace lucius
{

namespace ir
{

/*! \brief The implementation of a class that represents an operation in the program. */
class TargetControlOperationImplementation : public TargetOperationImplementation
{
public:
    using BasicBlockVector = std::vector<BasicBlock>;

public:
    virtual BasicBlockVector getPossibleTargets() const = 0;

};

} // namespace ir
} // namespace lucius

