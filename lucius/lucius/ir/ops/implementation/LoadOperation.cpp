/*  \file   LoadOperation.cpp
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The source file for the LoadOperation class.
*/

// Lucius Includes
#include <lucius/ir/ops/interface/LoadOperation.h>

#include <lucius/ir/interface/ShapeList.h>
#include <lucius/ir/interface/Shape.h>
#include <lucius/ir/interface/Value.h>
#include <lucius/ir/interface/Variable.h>
#include <lucius/ir/interface/Use.h>

#include <lucius/ir/implementation/OperationImplementation.h>

namespace lucius
{
namespace ir
{

class LoadOperationImplementation : public OperationImplementation
{
public:
    LoadOperationImplementation()
    {

    }

    ShapeList getOutputShapes(const ShapeList& inputShapes) const
    {
        return {inputShapes.front()};
    }

    ShapeList getInputShapes(const ShapeList& outputShapes) const
    {
        return {outputShapes.front()};
    }

public:
    virtual std::shared_ptr<ValueImplementation> clone() const
    {
        return std::make_shared<LoadOperationImplementation>(*this);
    }

public:
    std::string name() const
    {
        return "load";
    }

public:
    Type getType() const
    {
        return getOperand(0).getValue().getType();
    }

public:
    virtual bool isLoad() const
    {
        return true;
    }

};

LoadOperation::LoadOperation(Variable newValue)
: Operation(std::make_shared<LoadOperationImplementation>())
{
    setOperands({newValue});
}

LoadOperation::~LoadOperation()
{
    // intentionally blank
}

} // namespace ir
} // namespace lucius




