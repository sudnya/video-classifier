/*  \file   ConstantOperators.cpp
    \author Gregory Diamos
    \date   October 11, 2017
    \brief  The source file for the ConstantOperators class.
*/

// Lucius Includes
#include <lucius/ir/values/interface/ConstantOperator.h>

#include <lucius/ir/interface/Use.h>

#include <lucius/ir/implementation/ConstantImplementation.h>

// Standard Library Includes
#include <string>

namespace lucius
{

namespace ir
{

enum OperatorIds
{
    Unknown = 0,
    AddId = 1,
    MultiplyId = 2
};

class ConstantOperatorValueImplementation : public ConstantImplementation
{
public:
    ConstantOperatorValueImplementation(size_t id)
    : _operatorId(id)
    {

    }

    size_t getOperatorId() const
    {
        return _operatorId;
    }

public:
    virtual std::shared_ptr<ValueImplementation> clone() const
    {
        return std::make_shared<ConstantOperatorValueImplementation>(*this);
    }

public:
    std::string toString() const
    {
        if(getOperatorId() == AddId)
        {
            return "AddOperator";
        }
        else if(getOperatorId() == MultiplyId)
        {
            return "MultiplyOperator";
        }

        return "UnknownOperator";
    }

public:
    Type getType() const
    {
        return Type(Type::IntegerId);
    }

private:
    size_t _operatorId;
};

ConstantOperator::ConstantOperator(size_t operatorId)
: Constant(std::make_shared<ConstantOperatorValueImplementation>(operatorId))
{

}

size_t ConstantOperator::getOperatorId() const
{
    return std::static_pointer_cast<ConstantOperatorValueImplementation>(
        getValueImplementation())->getOperatorId();
}

Add::Add()
: ConstantOperator(AddId)
{

}

Multiply::Multiply()
: ConstantOperator(MultiplyId)
{

}

} // namespace ir
} // namespace lucius









