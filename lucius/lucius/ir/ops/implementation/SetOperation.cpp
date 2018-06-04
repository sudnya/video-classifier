/*  \file   SetOperation.cpp
    \author Gregory Diamos
    \date   October 1, 2017
    \brief  The source file for the SetOperation class.
*/

// Lucius Includes
#include <lucius/ir/ops/interface/SetOperation.h>

#include <lucius/ir/interface/ShapeList.h>
#include <lucius/ir/interface/Value.h>
#include <lucius/ir/interface/Use.h>
#include <lucius/ir/interface/Shape.h>

#include <lucius/ir/values/interface/ConstantInteger.h>

#include <lucius/matrix/interface/Precision.h>

#include <lucius/ir/types/interface/TensorType.h>
#include <lucius/ir/types/interface/StructureType.h>

#include <lucius/ir/implementation/OperationImplementation.h>

#include <lucius/util/interface/debug.h>

// Standard Library Includes
#include <cassert>

namespace lucius
{

namespace ir
{

class SetOperationImplementation : public OperationImplementation
{

public:
    ShapeList getOutputShapes(const ShapeList& inputShapes) const
    {
        auto dimensionsValue = getOperand(1).getValue();

        Shape outputShape;

        if(dimensionsValue.isConstant())
        {
            outputShape = value_cast<ConstantShape>(dimensionsValue).getContents();
        }
        else
        {
            outputShape.setUnknown();
        }

        return {outputShape};
    }

    ShapeList getInputShapes(const ShapeList& outputShapes) const
    {
        return {{1}};
    }

public:
    virtual std::shared_ptr<ValueImplementation> clone() const
    {
        return std::make_shared<SetOperationImplementation>(*this);
    }

public:
    std::string name() const
    {
        return "set";
    }

public:
    static matrix::Precision convertToPrecision(const Type& type)
    {
        if(type.isFloat())
        {
            return matrix::SinglePrecision();
        }
        else if(type.isDouble())
        {
            return matrix::DoublePrecision();
        }

        assert(type.isHalf());

        return matrix::HalfPrecision();
    }

    Type getType() const
    {
        auto valueType = getOperand(0).getValue().getType();
        auto shapeType = type_cast<ShapeType>(getOperand(1).getValue().getType());

        return TensorType(convertToPrecision(valueType), shapeType.getShape());
    }
};

SetOperation::SetOperation(Value value, Value shape)
: Operation(std::make_shared<SetOperationImplementation>())
{
    setOperands({value, position});
}

SetOperation::~SetOperation()
{

}

} // namespace ir
} // namespace lucius







