/*  \file   ReduceOperation.cpp
    \author Gregory Diamos
    \date   October 9, 2017
    \brief  The source file for the ReduceOperation class.
*/

// Lucius Includes
#include <lucius/ir/ops/interface/ReduceOperation.h>

#include <lucius/ir/values/interface/ConstantTensor.h>

#include <lucius/ir/interface/ShapeList.h>
#include <lucius/ir/interface/Shape.h>
#include <lucius/ir/interface/Use.h>
#include <lucius/ir/interface/Value.h>

#include <lucius/ir/implementation/OperationImplementation.h>

#include <lucius/matrix/interface/Matrix.h>

namespace lucius
{

namespace ir
{

class ReduceOperationImplementation : public OperationImplementation
{
public:
    ShapeList getOutputShapes(const ShapeList& inputShapes) const
    {
        auto dimensionsValue = getOperand(2).getValue();

        auto outputShape = inputShapes.front();

        if(dimensionsValue.isConstant())
        {
            auto dimensions = value_cast<ConstantTensor>(dimensionsValue).getContents();

            for(auto index : dimensions)
            {
                outputShape[index] = 1;
            }
        }
        else
        {
            outputShape.setUnknown();
        }

        return {outputShape};
    }

    ShapeList getInputShapes(const ShapeList& outputShapes) const
    {
        return {getOperandShape(0)};
    }

};

ReduceOperation::ReduceOperation(Value input, Value dimensions, Value operation)
: Operation(std::make_shared<ReduceOperationImplementation>())
{
    setOperands({input, dimensions, operation});
}

ReduceOperation::~ReduceOperation()
{
    // intentionally blank
}

} // namespace ir
} // namespace lucius







