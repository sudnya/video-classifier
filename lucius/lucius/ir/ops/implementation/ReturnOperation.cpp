/*  \file   ReturnOperation.cpp
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The source file for the ReturnOperation class.
*/

// Lucius Includes
#include <lucius/ir/ops/interface/ReturnOperation.h>

#include <lucius/ir/interface/ShapeList.h>
#include <lucius/ir/interface/Shape.h>
#include <lucius/ir/interface/Value.h>
#include <lucius/ir/interface/BasicBlock.h>
#include <lucius/ir/interface/Use.h>

#include <lucius/ir/types/interface/VoidType.h>

#include <lucius/ir/ops/implementation/ControlOperationImplementation.h>

// Standard Library Includes
#include <sstream>

namespace lucius
{

namespace ir
{

class ReturnOperationImplementation : public ControlOperationImplementation
{
public:
    ShapeList getOutputShapes(const ShapeList& inputShapes) const
    {
        return ShapeList();
    }

    ShapeList getInputShapes(const ShapeList& outputShapes) const
    {
        return {getOperandShape(0)};
    }

public:
    virtual std::shared_ptr<ValueImplementation> clone() const
    {
        return std::make_shared<ReturnOperationImplementation>(*this);
    }

public:
    std::string name() const
    {
        return "return";
    }

public:
    Type getType() const
    {
        if(empty())
        {
            return VoidType();
        }

        return getOperand(0).getValue().getType();
    }

public:
    bool isReturn() const
    {
        return true;
    }

public:
    std::string toString() const
    {
        std::stringstream stream;

        stream << name() << " ";

        auto operandIterator = getOperands().begin();

        if(operandIterator != getOperands().end())
        {
            auto& operand = *operandIterator;

            stream << operand.getValue().toSummaryString();

            ++operandIterator;
        }

        for( ; operandIterator != getOperands().end(); ++operandIterator)
        {
            auto& operand = *operandIterator;

            stream << ", " << operand.getValue().toSummaryString();
        }

        return stream.str();
    }

    virtual BasicBlockVector getPossibleTargets() const
    {
        return {};
    }

    virtual bool canFallthrough() const
    {
        return false;
    }

};

ReturnOperation::ReturnOperation()
: ControlOperation(std::make_shared<ReturnOperationImplementation>())
{

}

ReturnOperation::ReturnOperation(Value returnedValue)
: ControlOperation(std::make_shared<ReturnOperationImplementation>())
{
    setOperands({returnedValue});
}

ReturnOperation::~ReturnOperation()
{
    // intentionally blank
}

} // namespace ir
} // namespace lucius





