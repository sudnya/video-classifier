/*  \file   PHIOperationImplementation.cpp
    \author Gregory Diamos
    \date   October 12, 2017
    \brief  The source file for the PHIOperationImplementation class.
*/

// Lucius Includes
#include <lucius/ir/ops/implementation/PHIOperationImplementation.h>

#include <lucius/ir/interface/Use.h>
#include <lucius/ir/interface/Value.h>

// Standard Library Includes
#include <string>

namespace lucius
{

namespace ir
{

std::shared_ptr<ValueImplementation> PHIOperationImplementation::clone() const
{
    return std::make_shared<PHIOperationImplementation>(*this);
}

std::string PHIOperationImplementation::name() const
{
    return "phi";
}

Type PHIOperationImplementation::getType() const
{
    assert(!empty());

    return getOperands().front().getValue().getType();
}

} // namespace ir
} // namespace lucius







