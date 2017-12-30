/*  \file   ReturnOperation.cpp
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The source file for the ReturnOperation class.
*/

// Lucius Includes
#include <lucius/ir/ops/interface/ReturnOperation.h>

#include <lucius/ir/interface/ShapeList.h>
#include <lucius/ir/interface/Shape.h>
#include <lucius/ir/interface/Value.h>
#include <lucius/ir/interface/Use.h>

#include <lucius/ir/types/interface/VoidType.h>

#include <lucius/ir/implementation/OperationImplementation.h>

namespace lucius
{

namespace ir
{

class ReturnOperationImplementation : public OperationImplementation
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

public:
    virtual std::shared_ptr<ValueImplementation> clone() const
    {
        return std::make_shared<ReturnOperationImplementation>(*this);
    }

public:
    std::string name() const
    {
        return "return";
    }

public:
    Type getType() const
    {
        return VoidType();
    }

};

ReturnOperation::ReturnOperation(Value returnedValue)
: ControlOperation(std::make_shared<ReturnOperationImplementation>())
{
    setOperands({returnedValue});
}

ReturnOperation::~ReturnOperation()
{
    // intentionally blank
}

} // namespace ir
} // namespace lucius





