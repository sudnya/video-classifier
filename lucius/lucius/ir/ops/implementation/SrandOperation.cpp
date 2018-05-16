/*  \file   SrandOperation.cpp
    \author Gregory Diamos
    \date   October 9, 2017
    \brief  The source file for the SrandOperation class.
*/

// Lucius Includes
#include <lucius/ir/ops/interface/SrandOperation.h>

#include <lucius/ir/interface/ShapeList.h>
#include <lucius/ir/interface/Shape.h>
#include <lucius/ir/interface/Use.h>
#include <lucius/ir/interface/Value.h>

#include <lucius/ir/implementation/OperationImplementation.h>

#include <lucius/ir/types/interface/RandomStateType.h>

namespace lucius
{

namespace ir
{

class SrandOperationImplementation : public OperationImplementation
{
public:
    ShapeList getOutputShapes(const ShapeList& inputShapes) const
    {
        return {};
    }

    ShapeList getInputShapes(const ShapeList& outputShapes) const
    {
        return ShapeList({getOperandShape(0)});
    }

public:
    virtual std::shared_ptr<ValueImplementation> clone() const
    {
        return std::make_shared<SrandOperationImplementation>(*this);
    }

public:
    std::string name() const
    {
        return "srand";
    }

public:
    Type getType() const
    {
        return RandomStateType();
    }


};

SrandOperation::SrandOperation(Value seed)
: Operation(std::make_shared<SrandOperationImplementation>())
{
    setOperands({seed});
}

SrandOperation::~SrandOperation()
{
    // intentionally blank
}

} // namespace ir
} // namespace lucius




