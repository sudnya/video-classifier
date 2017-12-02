/*  \file   ZerosOperation.cpp
    \author Gregory Diamos
    \date   October 11, 2017
    \brief  The source file for the ZerosOperation class.
*/

// Lucius Includes
#include <lucius/ir/ops/interface/ZerosOperation.h>

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

class ZerosOperationImplementation : public OperationImplementation
{
public:
    ZerosOperationImplementation(const Type& tensorType)
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
        return std::make_shared<ZerosOperationImplementation>(*this);
    }

public:
    std::string name() const
    {
        return "zeros";
    }

private:
    Type _tensorType;

};


ZerosOperation::ZerosOperation(Type tensorType)
: Operation(std::make_shared<ZerosOperationImplementation>(tensorType))
{

}

ZerosOperation::~ZerosOperation()
{
    // intentionally blank
}

} // namespace ir
} // namespace lucius




