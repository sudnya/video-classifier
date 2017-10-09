/*  \file   BroadcastOperation.h
    \author Gregory Diamos
    \date   October 2, 2017
    \brief  The source file for the BroadcastOperation class.
*/

// Lucius Includes
#include <lucius/ir/ops/interface/BroadcastOperation.h>

#include <lucius/ir/values/interface/ConstantTensor.h>

#include <lucius/ir/interface/ShapeList.h>
#include <lucius/ir/interface/Shape.h>
#include <lucius/ir/interface/Use.h>

#include <lucius/ir/implementation/OperationImplementation.h>

#include <lucius/matrix/interface/Matrix.h>

namespace lucius
{

namespace ir
{

class BroadcastOperationImplementation : public OperationImplementation
{

public:
    ShapeList getOutputShapes(const ShapeList& inputShapes) const
    {
        return {inputShapes.front()};
    }

    ShapeList getInputShapes(const ShapeList& outputShapes) const
    {
        auto dimensionsValue = getOperand(2).getValue();

        auto rightShape = outputShapes.front();

        if(dimensionsValue.isConstant())
        {
            auto dimensions = value_cast<ConstantTensor>(dimensionsValue).getContents();

            for(auto index : dimensions)
            {
                rightShape[index] = 1;
            }
        }
        else
        {
            rightShape.setUnknown();
        }

        return {outputShapes.front(), rightShape};
    }

};

BroadcastOperation::BroadcastOperation(Value left, Value right, Value dimensions, Value operation)
: Operation(std::make_shared<BroadcastOperationImplementation>())
{
    setOperands({left, right, dimensions, operation});
}

BroadcastOperation::~BroadcastOperation()
{

}


} // namespace ir
} // namespace lucius








