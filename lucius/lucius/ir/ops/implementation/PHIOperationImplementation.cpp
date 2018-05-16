/*  \file   PHIOperationImplementation.cpp
    \author Gregory Diamos
    \date   October 12, 2017
    \brief  The source file for the PHIOperationImplementation class.
*/

// Lucius Includes
#include <lucius/ir/ops/implementation/PHIOperationImplementation.h>

#include <lucius/ir/interface/Use.h>
#include <lucius/ir/interface/Value.h>
#include <lucius/ir/interface/BasicBlock.h>

// Standard Library Includes
#include <string>
#include <sstream>

namespace lucius
{

namespace ir
{

PHIOperationImplementation::PHIOperationImplementation(const Type& type)
: _type(type)
{

}

std::shared_ptr<ValueImplementation> PHIOperationImplementation::clone() const
{
    return std::make_shared<PHIOperationImplementation>(*this);
}

std::string PHIOperationImplementation::name() const
{
    return "phi";
}

std::string PHIOperationImplementation::toString() const
{
    assert(size() == _incomingBlocks.size());

    std::stringstream stream;

    stream << "%" << getId() << " = " << name() << " ";

    auto operandIterator = getOperands().begin();
    auto incomingBlockIterator = _incomingBlocks.begin();

    if(operandIterator != getOperands().end())
    {
        auto& operand = *operandIterator;

        stream << operand.getValue().toSummaryString() << " <- "
            << incomingBlockIterator->name();

        ++operandIterator;
        ++incomingBlockIterator;
    }

    for( ; operandIterator != getOperands().end(); ++operandIterator, ++incomingBlockIterator)
    {
        auto& operand = *operandIterator;

        stream << ", " << operand.getValue().toSummaryString() << " <- "
            << incomingBlockIterator->name();
    }

    return stream.str();
}

Type PHIOperationImplementation::getType() const
{
    return _type;
}

bool PHIOperationImplementation::isPHI() const
{
    return true;
}

const PHIOperationImplementation::BasicBlockVector&
    PHIOperationImplementation::getIncomingBasicBlocks() const
{
    return _incomingBlocks;
}

void PHIOperationImplementation::addIncomingValue(const Value& value,
    const BasicBlock& incoming)
{
    appendOperand(value);
    _incomingBlocks.push_back(incoming);
}

} // namespace ir
} // namespace lucius







