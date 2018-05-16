/*  \file   StoreOperation.cpp
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the StoreOperation class.
*/

// Lucius Includes
#include <lucius/ir/ops/interface/StoreOperation.h>

#include <lucius/ir/interface/ShapeList.h>
#include <lucius/ir/interface/Shape.h>
#include <lucius/ir/interface/Use.h>
#include <lucius/ir/interface/Value.h>
#include <lucius/ir/interface/Variable.h>

#include <lucius/ir/types/interface/VoidType.h>

#include <lucius/ir/implementation/OperationImplementation.h>

namespace lucius
{

namespace ir
{

class StoreOperationImplementation : public OperationImplementation
{
public:
    ShapeList getOutputShapes(const ShapeList& inputShapes) const
    {
        return {};
    }

    ShapeList getInputShapes(const ShapeList& outputShapes) const
    {
        return {};
    }

public:
    virtual std::shared_ptr<ValueImplementation> clone() const
    {
        return std::make_shared<StoreOperationImplementation>(*this);
    }

public:
    std::string name() const
    {
        return "store";
    }

public:
    Type getType() const
    {
        return VoidType();
    }


};

StoreOperation::StoreOperation(Variable variable, Value input)
: Operation(std::make_shared<StoreOperationImplementation>())
{
    setOperands({variable, input});
}

StoreOperation::~StoreOperation()
{
    // intentionally blank
}

} // namespace ir
} // namespace lucius



