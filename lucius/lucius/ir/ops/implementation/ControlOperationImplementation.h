/*  \file   ControlOperationImplementation.h
    \author Gregory Diamos
    \date   October 12, 2017
    \brief  The header file for the ControlOperationImplementation class.
*/

#pragma once

// Lucius Includes
#include <lucius/ir/implementation/OperationImplementation.h>

namespace lucius
{

namespace ir
{

class ControlOperationImplementation : public OperationImplementation
{
public:
    virtual std::string name() const;

public:
    virtual Type getType() const;

public:
    using BasicBlockVector = std::vector<BasicBlock>;

public:
    virtual BasicBlockVector getPossibleTargets() const = 0;

public:
    virtual bool canFallthrough() const = 0;

};

} // namespace ir
} // namespace lucius





