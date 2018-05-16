/*  \file   OperatorData.cpp
    \author Gregory Diamos
    \date   October 1, 2017
    \brief  The source file for the OperatorData class.
*/

// Lucius Includes
#include <lucius/machine/generic/interface/OperatorData.h>

#include <lucius/ir/target/implementation/TargetValueDataImplementation.h>

#include <lucius/matrix/interface/Operator.h>

// Standard Library Includes
#include <cassert>

namespace lucius
{
namespace machine
{
namespace generic
{

class OperatorDataImplementation : public ir::TargetValueDataImplementation
{
public:
    matrix::Operator getOperator() const
    {
        return _operator;
    }

public:
    virtual void* getData() const
    {
        return const_cast<void*>(reinterpret_cast<const void*>(&_operator));
    }

private:
   matrix::Operator _operator;

};

OperatorData::OperatorData()
: OperatorData(std::make_shared<OperatorDataImplementation>())
{

}

OperatorData::OperatorData(std::shared_ptr<ir::TargetValueDataImplementation> implementation)
: _implementation(std::static_pointer_cast<OperatorDataImplementation>(implementation))
{
    assert(static_cast<bool>(std::dynamic_pointer_cast<OperatorDataImplementation>(
        implementation)));
}

matrix::Operator OperatorData::getOperator() const
{
    return _implementation->getOperator();
}

std::shared_ptr<ir::TargetValueDataImplementation> OperatorData::getImplementation() const
{
    return _implementation;
}

} // namespace generic
} // namespace machine
} // namespace lucius










