/*  \file   Gradient.cpp
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The source file for the Gradient class.
*/

// Lucius Includes
#include <lucius/ir/interface/Gradient.h>

#include <lucius/ir/interface/Value.h>
#include <lucius/ir/interface/Operation.h>
#include <lucius/ir/interface/Use.h>

#include <lucius/ir/implementation/OperationImplementation.h>

namespace lucius
{

namespace ir
{

class GradientOperationImplementation : public OperationImplementation
{
public:
    std::shared_ptr<ValueImplementation> clone() const
    {
        return std::make_shared<GradientOperationImplementation>(*this);
    }

public:
    std::string name() const
    {
        return "gradient";
    }
};

Gradient::Gradient()
: _implementation(std::make_shared<GradientOperationImplementation>())
{

}

Gradient::Gradient(Value v)
: _implementation(std::dynamic_pointer_cast<GradientOperationImplementation>(
    v.getValueImplementation()))
{

}

Gradient::Gradient(Operation o)
: _implementation(std::dynamic_pointer_cast<GradientOperationImplementation>(
    o.getValueImplementation()))
{

}

Gradient::~Gradient()
{
    // intentionally blank
}

std::shared_ptr<ValueImplementation> Gradient::getValueImplementation()
{
    return _implementation;
}

} // namespace ir
} // namespace lucius






