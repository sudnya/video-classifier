/*  \file   PHIOperationImplementation.h
    \author Gregory Diamos
    \date   October 12, 2017
    \brief  The header file for the PHIOperationImplementation class.
*/

#pragma once

// Lucius Includes
#include <lucius/ir/implementation/OperationImplementation.h>

namespace lucius
{

namespace ir
{

class PHIOperationImplementation : public OperationImplementation
{
public:
    virtual std::shared_ptr<ValueImplementation> clone() const;

public:
    virtual std::string name() const;

public:
    virtual Type getType() const;

};

} // namespace ir
} // namespace lucius






