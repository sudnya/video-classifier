/*  \file   GetOperation.cpp
    \author Gregory Diamos
    \date   October 1, 2017
    \brief  The source file for the GetOperation class.
*/

// Lucius Includes
#include <lucius/ir/ops/interface/GetOperation.h>

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

class GetOperationImplementation : public OperationImplementation
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

public:
    virtual std::shared_ptr<ValueImplementation> clone() const
    {
        return std::make_shared<GetOperationImplementation>(*this);
    }

public:
    std::string name() const
    {
        return "get";
    }

public:
    Type getType() const
    {
        auto operandType = getOperand(0).getValue().getType();

        if(operandType.isTensor())
        {
            auto tensorType = type_cast<TensorType>(operandType);

            if(tensorType.getPrecision() == matrix::SinglePrecision())
            {
                return Type(Type::FloatId);
            }
            else if(tensorType.getPrecision() == matrix::HalfPrecision())
            {
                return Type(Type::HalfId);
            }

            assert(tensorType.getPrecision() == matrix::DoublePrecision());

            return Type(Type::DoubleId);
        }
        else
        {
            // TODO: handle more diverse types
            assert(operandType.isStructure());

            auto indexOperand = getOperand(1).getValue();

            assert(indexOperand.isConstant());
            assert(indexOperand.isInteger());

            auto constantInteger = value_cast<ConstantInteger>(indexOperand);

            size_t index = constantInteger.getValue();

            auto structure = type_cast<StructureType>(operandType);

            assert(index < structure.size());

            return structure[index];
        }
    }
};

GetOperation::GetOperation(Value container, Value position)
: Operation(std::make_shared<GetOperationImplementation>())
{
    setOperands({container, position});
}

GetOperation::~GetOperation()
{

}

} // namespace ir
} // namespace lucius






