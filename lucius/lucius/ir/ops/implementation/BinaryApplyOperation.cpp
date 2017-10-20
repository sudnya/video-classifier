/*  \file   BinaryApplyOperation.h
    \author Gregory Diamos
    \date   October 1, 2017
    \brief  The source file for the BinaryApplyOperation class.
*/

// Lucius Includes
#include <lucius/ir/ops/interface/BinaryApplyOperation.h>

#include <lucius/ir/interface/ShapeList.h>
#include <lucius/ir/interface/Shape.h>
#include <lucius/ir/interface/Value.h>
#include <lucius/ir/interface/Use.h>

#include <lucius/ir/implementation/OperationImplementation.h>

namespace lucius
{

namespace ir
{

class BinaryApplyOperationImplementation : public OperationImplementation
{

public:
    ShapeList getOutputShapes(const ShapeList& inputShapes) const
    {
        return {inputShapes.front()};
    }

    ShapeList getInputShapes(const ShapeList& outputShapes) const
    {
        return {outputShapes.front(), outputShapes.front()};
    }

};

BinaryApplyOperation::BinaryApplyOperation(Value left, Value right, Value operation)
: Operation(std::make_shared<BinaryApplyOperationImplementation>())
{
    setOperands({left, right, operation});
}

BinaryApplyOperation::~BinaryApplyOperation()
{
    // intentionally blank
}

} // namespace ir
} // namespace lucius






