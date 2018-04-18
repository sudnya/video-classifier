/*  \file   ControlOperation.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the ControlOperation class.
*/

#pragma once

// Lucius Includes
#include <lucius/ir/interface/Operation.h>

// Standard Library Includes
#include <vector>

namespace lucius
{

namespace ir
{

/*! \brief A class for representing a branch. */
class ControlOperation : public Operation
{
public:
    explicit ControlOperation(std::shared_ptr<ValueImplementation>);
    ~ControlOperation();

public:
    using BasicBlockVector = std::vector<BasicBlock>;

public:
    BasicBlockVector getPossibleTargets() const;

public:
    bool canFallthrough() const;

};

} // namespace ir
} // namespace lucius




