/*  \file   FloatValue.cpp
    \author Gregory Diamos
    \date   October 11, 2017
    \brief  The source file for the FloatValue class.
*/

// Lucius Includes
#include <lucius/machine/generic/interface/FloatValue.h>

#include <lucius/machine/generic/interface/FloatData.h>

#include <lucius/ir/interface/Use.h>

#include <lucius/ir/target/implementation/TargetValueImplementation.h>

// Standard Library Include
#include <string>

namespace lucius
{
namespace machine
{
namespace generic
{

class FloatValueImplementation : public ir::TargetValueImplementation
{
public:
    using Type = ir::Type;
    using Context = ir::Context;
    using TargetValueData = ir::TargetValueData;

public:
    FloatValueImplementation(Context& context)
    : _type(Type::FloatId)
    {
        bindToContext(&context);
    }

public:
    virtual std::shared_ptr<ValueImplementation> clone() const
    {
        return std::make_shared<FloatValueImplementation>(*this);
    }

    std::string toString() const
    {
        return getType().toString() + " " + "%" + std::to_string(getId());
    }

public:
    Type getType() const
    {
        return _type;
    }

public:
    void allocateData()
    {
        setData(FloatData(0));
    }

    void freeData()
    {
        setData(TargetValueData());
    }

private:
    Type _type;
};

FloatValue::FloatValue( Context& context)
: FloatValue(std::make_shared<FloatValueImplementation>(context))
{

}

FloatValue::FloatValue(std::shared_ptr<ir::ValueImplementation> implementation)
: TargetValue(implementation)
{

}

} // namespace generic
} // namespace machine
} // namespace lucius











