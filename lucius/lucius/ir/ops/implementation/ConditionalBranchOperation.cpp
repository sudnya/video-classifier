/*  \file   ConditionalBranchOperation.cpp
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The source file for the ConditionalBranchOperation class.
*/

// Lucius Includes
#include <lucius/ir/ops/interface/ConditionalBranchOperation.h>

#include <lucius/ir/interface/ShapeList.h>
#include <lucius/ir/interface/Shape.h>
#include <lucius/ir/interface/Use.h>
#include <lucius/ir/interface/Value.h>

#include <lucius/ir/interface/BasicBlock.h>

#include <lucius/ir/ops/implementation/ControlOperationImplementation.h>

namespace lucius
{

namespace ir
{

class ConditionalBranchOperationImplementation : public ControlOperationImplementation
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
        return std::make_shared<ConditionalBranchOperationImplementation>(*this);
    }

public:
    std::string name() const
    {
        return "branch";
    }

public:
    Type getType() const
    {
        return Type(Type::VoidId);
    }

    virtual BasicBlockVector getPossibleTargets() const
    {
        auto target      = value_cast<BasicBlock>(getOperand(1).getValue());
        auto fallthrough = value_cast<BasicBlock>(getOperand(2).getValue());

        return {target, fallthrough};
    }

    virtual bool canFallthrough() const
    {
        return false;
    }

};

ConditionalBranchOperation::ConditionalBranchOperation(Value predicate, BasicBlock target,
    BasicBlock fallthrough)
: ControlOperation(std::make_shared<ConditionalBranchOperationImplementation>())
{
    setOperands({predicate, target, fallthrough});
}

ConditionalBranchOperation::~ConditionalBranchOperation()
{
    // intentionally blank
}

} // namespace ir
} // namespace lucius



