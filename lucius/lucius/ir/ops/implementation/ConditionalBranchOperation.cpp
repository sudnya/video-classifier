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

#include <lucius/ir/implementation/OperationImplementation.h>

namespace lucius
{

namespace ir
{

class ConditionalBranchOperationImplementation : public OperationImplementation
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

};

ConditionalBranchOperation::ConditionalBranchOperation(Value predicate, BasicBlock target,
    BasicBlock fallthrough)
: ControlOperation(std::make_shared<ConditionalBranchOperationImplementation>())
{
    setOperands({predicate, target, fallthrough});
}

ConditionalBranchOperation::~ConditionalBranchOperation()
{

}

} // namespace ir
} // namespace lucius



