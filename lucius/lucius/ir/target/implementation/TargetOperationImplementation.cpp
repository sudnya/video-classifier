/*  \file   TargetOperationImplementation.cpp
    \author Gregory Diamos
    \date   October 4, 2017
    \brief  The source file for the TargetOperationImplementation class.
*/

// Lucius Includes
#include <lucius/ir/target/implementation/TargetOperationImplementation.h>

#include <lucius/ir/interface/Use.h>
#include <lucius/ir/interface/Value.h>

#include <lucius/matrix/interface/Matrix.h>
#include <lucius/matrix/interface/GenericOperators.h>
#include <lucius/matrix/interface/Operator.h>

#include <lucius/ir/target/interface/TargetValue.h>
#include <lucius/ir/target/interface/TargetValueData.h>
#include <lucius/ir/target/interface/TensorData.h>
#include <lucius/ir/target/interface/OperatorData.h>

#include <lucius/ir/values/interface/ConstantTensor.h>
#include <lucius/ir/values/interface/ConstantOperator.h>

// Standard Library Includes
#include <cassert>
#include <string>
#include <sstream>

namespace lucius
{

namespace ir
{

using UseList = std::list<Use>;

TargetOperationImplementation::TargetOperationImplementation()
: _hasOutputOperand(false)
{

}

Use& TargetOperationImplementation::getOutputOperand()
{
    return getOperands().back();
}

const Use& TargetOperationImplementation::getOutputOperand() const
{
    return getOperands().back();
}

TargetOperationImplementation::iterator TargetOperationImplementation::getOutputOperandPosition()
{
    return --getOperands().end();
}

TargetOperationImplementation::const_iterator
    TargetOperationImplementation::getOutputOperandPosition() const
{
    return --getOperands().end();
}

bool TargetOperationImplementation::hasOutputOperand() const
{
    return _hasOutputOperand;
}

void TargetOperationImplementation::setOutputOperand(const TargetValue& v)
{
    if(!_hasOutputOperand)
    {
        getOperands().push_back(Use(v.getValue()));
        _hasOutputOperand = true;
    }
    else
    {
        getOutputOperand() = Use(v.getValue());
    }
}

void TargetOperationImplementation::setOperand(const TargetValue& v, size_t index)
{
    _growToSupportIndex(index);

    getOperand(index) = Use(v.getValue());
}

void TargetOperationImplementation::appendOperand(const TargetValue& v)
{
    auto end = getOperands().end();

    if(_hasOutputOperand)
    {
        assert(!getOperands().empty());
        --end;
    }

    getOperands().insert(end, Use(v.getValue()));
}

matrix::Matrix TargetOperationImplementation::getOperandDataAsTensor(size_t index) const
{
    auto operand = getOperand(index);

    auto value = ir::value_cast<TargetValue>(operand.getValue());

    return value.getDataAsTensor();
}

matrix::Operator TargetOperationImplementation::getOperandDataAsOperator(size_t index) const
{
    auto operand = getOperand(index);

    auto value = ir::value_cast<TargetValue>(operand.getValue());

    return value.getDataAsOperator();

    if(value.isConstant())
    {
        auto constant = ir::value_cast<ConstantOperator>(value);

        return constant.getOperator();
    }
    else
    {
        auto data = data_cast<OperatorData>(value.getData());

        return data.getOperator();
    }
}

std::string TargetOperationImplementation::toString() const
{
    std::stringstream stream;

    if(hasOutputOperand())
    {
        stream << getOutputOperand().toString() << " = ";
    }

    stream << name();

    if(!empty())
    {
        stream << " ";

        auto operandIterator = getOperands().begin();
        auto end = getOperands().end();

        if(hasOutputOperand())
        {
            --end;
        }

        if(operandIterator != end)
        {
            auto& operand = *operandIterator;

            stream << operand.getValue().toSummaryString();

            ++operandIterator;
        }

        for( ; operandIterator != end; ++operandIterator)
        {
            auto& operand = *operandIterator;

            stream << ", " << operand.getValue().toSummaryString();
        }
    }

    return stream.str();
}

void TargetOperationImplementation::_growToSupportIndex(size_t index)
{
    size_t size = getOperands().size();

    if(_hasOutputOperand)
    {
        --size;
    }

    for(size_t i = size; i < index; ++i)
    {
        getOperands().push_back(Use());
    }
}

} // namespace ir
} // namespace lucius







