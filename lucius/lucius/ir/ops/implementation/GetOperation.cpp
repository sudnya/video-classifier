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

#include <lucius/ir/types/interface/TensorType.h>

#include <lucius/ir/implementation/OperationImplementation.h>

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






