/*  \file   LessThanOperation.cpp
    \author Gregory Diamos
    \date   October 1, 2017
    \brief  The source file for the LessThanOperation class.
*/

// Lucius Includes
#include <lucius/ir/ops/interface/LessThanOperation.h>

#include <lucius/ir/interface/ShapeList.h>
#include <lucius/ir/interface/Value.h>
#include <lucius/ir/interface/Use.h>
#include <lucius/ir/interface/Shape.h>

#include <lucius/ir/types/interface/TensorType.h>

#include <lucius/ir/implementation/OperationImplementation.h>

// Standard Library Includes
#include <cassert>

namespace lucius
{

namespace ir
{

class LessThanOperationImplementation : public OperationImplementation
{

public:
    ShapeList getOutputShapes(const ShapeList& inputShapes) const
    {
        return {Shape({1})};
    }

    ShapeList getInputShapes(const ShapeList& outputShapes) const
    {
        return {getOperandShape(0), getOperandShape(1)};
    }
};

LessThanOperation::LessThanOperation(Value left, Value right)
: Operation(std::make_shared<LessThanOperationImplementation>())
{
    setOperands({left, right});
}

LessThanOperation::~LessThanOperation()
{

}

} // namespace ir
} // namespace lucius







