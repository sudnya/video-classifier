/*  \file   OnesOperation.cpp
    \author Gregory Diamos
    \date   October 4, 2017
    \brief  The source file for the OnesOperation class.
*/

// Lucius Includes
#include <lucius/ir/ops/interface/OnesOperation.h>

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

class OnesOperationImplementation : public OperationImplementation
{
public:
    OnesOperationImplementation(const Type& tensorType)
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
        return std::make_shared<OnesOperationImplementation>(*this);
    }

public:
    std::string name() const
    {
        return "ones";
    }

private:
    Type _tensorType;

};

OnesOperation::OnesOperation(Type tensorType)
: Operation(std::make_shared<OnesOperationImplementation>(tensorType))
{

}

OnesOperation::~OnesOperation()
{

}

} // namespace ir
} // namespace lucius



