/*  \file   RandOperation.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the RandOperation class.
*/

// Lucius Includes
#include <lucius/ir/ops/interface/RandOperation.h>

#include <lucius/ir/interface/ShapeList.h>
#include <lucius/ir/interface/Shape.h>

#include <lucius/ir/types/interface/TensorType.h>

#include <lucius/ir/implementation/OperationImplementation.h>

namespace lucius
{

namespace ir
{

class RandOperationImplementation : public OperationImplementation
{
public:
    RandOperationImplementation(const TensorType& tensorType)
    : _tensorType(tensorType)
    {

    }

public:
    ShapeList getOutputShapes(const ShapeList& inputShapes) const
    {
        return {getOperandShape(0), _tensorType.getShape()};
    }

    ShapeList getInputShapes(const ShapeList& outputShapes) const
    {
        return {getOperandShape(0)};
    }

private:
    TensorType _tensorType;

};

RandOperation::RandOperation(Value state, Type tensorType)
: Operation(std::make_shared<RandOperationImplementation>(type_cast<TensorType>(tensorType)))
{
    setOperands({state});
}

RandOperation::~RandOperation()
{
    // intentionally blank
}

} // namespace ir
} // namespace lucius





