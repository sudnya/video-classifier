/*  \file   OperatorData.cpp
    \author Gregory Diamos
    \date   October 1, 2017
    \brief  The source file for the OperatorData class.
*/

// Lucius Includes
#include <lucius/ir/target/interface/OperatorData.h>

#include <lucius/ir/target/implementation/TargetValueDataImplementation.h>

#include <lucius/matrix/interface/Operation.h>

namespace lucius
{

namespace ir
{

class OperatorDataImplementation : public TargetValueDataImplementation
{
public:
    matrix::Operation getOperator() const
    {
        return *_operator;
    }

public:
    virtual void* getData() const
    {
        return const_cast<void*>(reinterpret_cast<const void*>(&_operator));
    }

private:
    std::unique_ptr<matrix::Operation> _operator;

};

OperatorData::OperatorData()
: OperatorData(std::make_shared<OperatorDataImplementation>())
{

}

OperatorData::OperatorData(std::shared_ptr<TargetValueDataImplementation> implementation)
: _implementation(std::static_pointer_cast<OperatorDataImplementation>(implementation))
{
    assert(static_cast<bool>(std::dynamic_pointer_cast<OperatorDataImplementation>(
        implementation)));
}

matrix::Operation OperatorData::getOperator() const
{
    return _implementation->getOperator();
}

std::shared_ptr<TargetValueDataImplementation> OperatorData::getImplementation() const
{
    return _implementation;
}

} // namespace ir
} // namespace lucius










