/*  \file   ApplyOperation.cpp
    \author Gregory Diamos
    \date   October 1, 2017
    \brief  The source file for the ApplyOperation class.
*/

// Lucius Includes
#include <lucius/ir/ops/interface/ApplyOperation.h>

#include <lucius/ir/interface/ShapeList.h>
#include <lucius/ir/interface/Shape.h>
#include <lucius/ir/interface/Value.h>
#include <lucius/ir/interface/Use.h>

#include <lucius/ir/implementation/OperationImplementation.h>

namespace lucius
{

namespace ir
{

class ApplyOperationImplementation : public OperationImplementation
{

public:
    ShapeList getOutputShapes(const ShapeList& inputShapes) const
    {
        return inputShapes;
    }

    ShapeList getInputShapes(const ShapeList& outputShapes) const
    {
        return outputShapes;
    }

public:
    std::string name() const
    {
        return "apply";
    }

public:
    virtual std::shared_ptr<ValueImplementation> clone() const
    {
        return std::make_shared<ApplyOperationImplementation>(*this);
    }
};

ApplyOperation::ApplyOperation(Value input, Value operation)
: Operation(std::make_shared<ApplyOperationImplementation>())
{
    setOperands({input, operation});
}

ApplyOperation::~ApplyOperation()
{

}

} // namespace ir
} // namespace lucius





