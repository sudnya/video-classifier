/*  \file   RandnOperation.h
    \author Gregory Diamos
    \date   October 4, 2017
    \brief  The source file for the RandnOperation class.
*/

// Lucius Includes
#include <lucius/ir/ops/interface/RandnOperation.h>

#include <lucius/ir/interface/ShapeList.h>
#include <lucius/ir/interface/Shape.h>
#include <lucius/ir/interface/Use.h>
#include <lucius/ir/interface/Value.h>

#include <lucius/ir/types/interface/TensorType.h>

#include <lucius/ir/implementation/OperationImplementation.h>

namespace lucius
{

namespace ir
{

class RandnOperationImplementation : public OperationImplementation
{
public:
    RandnOperationImplementation(const TensorType& tensorType)
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

RandnOperation::RandnOperation(Value state, Type tensorType)
: Operation(std::make_shared<RandnOperationImplementation>(type_cast<TensorType>(tensorType)))
{
    setOperands({state});
}

RandnOperation::~RandnOperation()
{
    // intentionally blank
}

} // namespace ir
} // namespace lucius


