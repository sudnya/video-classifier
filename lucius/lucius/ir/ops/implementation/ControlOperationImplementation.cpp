/*  \file   ControlOperationImplementation.cpp
    \author Gregory Diamos
    \date   October 12, 2017
    \brief  The source file for the ControlOperationImplementation class.
*/

// Lucius Includes
#include <lucius/ir/ops/implementation/ControlOperationImplementation.h>

#include <lucius/ir/interface/Use.h>

// Standard Library Includes
#include <string>

namespace lucius
{

namespace ir
{

std::shared_ptr<ValueImplementation> ControlOperationImplementation::clone() const
{
    return std::make_shared<ControlOperationImplementation>(*this);
}

std::string ControlOperationImplementation::name() const
{
    return "control";
}

Type ControlOperationImplementation::getType() const
{
    return Type(Type::BasicBlockId);
}

} // namespace ir
} // namespace lucius






