/*  \file   Constant.cpp
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the Constant class.
*/

// Lucius Includes
#include <lucius/ir/interface/Constant.h>
#include <lucius/ir/implementation/ValueImplementation.h>

namespace lucius
{

namespace ir
{

class ConstantImplementation : public ValueImplementation
{
};

Constant::Constant(std::shared_ptr<ValueImplementation> implementation)
: _implementation(implementation)
{

}

std::shared_ptr<ValueImplementation> Constant::getValueImplementation() const
{
    return _implementation;
}

} // namespace ir
} // namespace lucius






