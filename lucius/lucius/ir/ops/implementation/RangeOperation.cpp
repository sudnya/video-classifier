/*  \file   RangeOperation.cpp
    \author Gregory Diamos
    \date   October 4, 2017
    \brief  The source file for the RangeOperation class.
*/

// Lucius Includes
#include <lucius/ir/ops/interface/RangeOperation.h>

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

class RangeOperationImplementation : public OperationImplementation
{
public:
    RangeOperationImplementation(const Type& tensorType)
    : _tensorType(tensorType)
    {

    }

    ShapeList getOutputShapes(const ShapeList& inputShapes) const
    {
        auto type = type_cast<TensorType>(_tensorType);

        return ShapeList({type.getShape()});
    }

    ShapeList getInputShapes(const ShapeList& outputShapes) const
    {
        return ShapeList();
    }

public:
    virtual std::shared_ptr<ValueImplementation> clone() const
    {
        return std::make_shared<RangeOperationImplementation>(*this);
    }

public:
    std::string name() const
    {
        return "range";
    }

private:
    Type _tensorType;

};

RangeOperation::RangeOperation(Type tensorType)
: Operation(std::make_shared<RangeOperationImplementation>(tensorType))
{

}

RangeOperation::~RangeOperation()
{
    // intetionally blank
}

} // namespace ir
} // namespace lucius




