/*  \file   ConstantOperators.cpp
    \author Gregory Diamos
    \date   October 11, 2017
    \brief  The source file for the ConstantOperators class.
*/

// Lucius Includes
#include <lucius/ir/values/interface/ConstantOperator.h>

#include <lucius/ir/interface/Use.h>

#include <lucius/matrix/interface/Operator.h>
#include <lucius/matrix/interface/GenericOperators.h>

#include <lucius/ir/implementation/ConstantImplementation.h>

// Standard Library Includes
#include <string>

namespace lucius
{

namespace ir
{

class ConstantOperatorValueImplementation : public ConstantImplementation
{
public:
    ConstantOperatorValueImplementation(const matrix::Operator& op)
    : _operator(op)
    {

    }

    const matrix::Operator& getOperator() const
    {
        return _operator;
    }

public:
    virtual std::shared_ptr<ValueImplementation> clone() const
    {
        return std::make_shared<ConstantOperatorValueImplementation>(*this);
    }

public:
    std::string toString() const
    {
        if(getOperator() == matrix::Add())
        {
            return "AddOperator";
        }
        else if(getOperator() == matrix::Multiply())
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
    matrix::Operator _operator;
};

ConstantOperator::ConstantOperator(const matrix::Operator& op)
: ConstantOperator(std::make_shared<ConstantOperatorValueImplementation>(op))
{

}

ConstantOperator::ConstantOperator(std::shared_ptr<ValueImplementation> i)
: Constant(i)
{

}

const matrix::Operator& ConstantOperator::getOperator() const
{
    return std::static_pointer_cast<ConstantOperatorValueImplementation>(
        getValueImplementation())->getOperator();
}

Add::Add()
: ConstantOperator(matrix::Add())
{

}

Multiply::Multiply()
: ConstantOperator(matrix::Multiply())
{

}

} // namespace ir
} // namespace lucius









