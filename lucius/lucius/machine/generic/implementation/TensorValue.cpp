/*  \file   TensorValue.cpp
    \author Gregory Diamos
    \date   October 11, 2017
    \brief  The source file for the TensorValue class.
*/

// Lucius Includes
#include <lucius/machine/generic/interface/TensorValue.h>

#include <lucius/ir/interface/Shape.h>
#include <lucius/ir/interface/Use.h>

#include <lucius/ir/target/interface/TensorData.h>

#include <lucius/ir/types/interface/TensorType.h>

#include <lucius/ir/target/implementation/TargetValueImplementation.h>

namespace lucius
{
namespace machine
{
namespace generic
{

class TensorValueImplementation : public ir::TargetValueImplementation
{
public:
    using Precision = TensorValue::Precision;
    using Shape = TensorValue::Shape;
    using TensorType = ir::TensorType;
    using Type = ir::Type;
    using Context = ir::Context;
    using TensorData = ir::TensorData;
    using TargetValueData = ir::TargetValueData;

public:
    TensorValueImplementation(const Shape& s, const Precision& p, Context& context)
    : _type(TensorType(s, p))
    {
        bindToContext(&context);
    }

public:
    const Shape& getShape() const
    {
        return _type.getShape();
    }

    const Precision& getPrecision() const
    {
        return _type.getPrecision();
    }

public:
    virtual std::shared_ptr<ValueImplementation> clone() const
    {
        return std::make_shared<TensorValueImplementation>(*this);
    }

    std::string toString() const
    {
        return "GenericTensor(" + getType().toString() + ")" + "%" + std::to_string(getId());
    }

public:
    Type getType() const
    {
        return _type;
    }

public:
    void allocateData()
    {
        setData(TensorData(getShape(), getPrecision()));
    }

    void freeData()
    {
        setData(TargetValueData());
    }

private:
    TensorType _type;
};

TensorValue::TensorValue(const Shape& s, const Precision& p, Context& context)
: TensorValue(std::make_shared<TensorValueImplementation>(s, p, context))
{

}

TensorValue::TensorValue(std::shared_ptr<ir::ValueImplementation> implementation)
: TargetValue(implementation)
{

}

const TensorValue::Shape& TensorValue::getShape() const
{
    return std::static_pointer_cast<TensorValueImplementation>(
        getValueImplementation())->getShape();
}

const TensorValue::Precision& TensorValue::getPrecision() const
{
    return std::static_pointer_cast<TensorValueImplementation>(
        getValueImplementation())->getPrecision();
}

} // namespace generic
} // namespace machine
} // namespace lucius









