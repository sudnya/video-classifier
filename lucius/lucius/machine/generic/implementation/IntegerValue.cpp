/*  \file   IntegerValue.cpp
    \author Gregory Diamos
    \date   October 11, 2017
    \brief  The source file for the IntegerValue class.
*/

// Lucius Includes
#include <lucius/machine/generic/interface/IntegerValue.h>

#include <lucius/machine/generic/interface/IntegerData.h>

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

class IntegerValueImplementation : public ir::TargetValueImplementation
{
public:
    using Type = ir::Type;
    using Context = ir::Context;
    using TargetValueData = ir::TargetValueData;

public:
    IntegerValueImplementation(Context& context)
    : _type(Type::IntegerId)
    {
        bindToContext(&context);
    }

public:
    virtual std::shared_ptr<ValueImplementation> clone() const
    {
        return std::make_shared<IntegerValueImplementation>(*this);
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
        setData(IntegerData(0));
    }

    void freeData()
    {
        setData(TargetValueData());
    }

private:
    Type _type;
};

IntegerValue::IntegerValue( Context& context)
: IntegerValue(std::make_shared<IntegerValueImplementation>(context))
{

}

IntegerValue::IntegerValue(std::shared_ptr<ir::ValueImplementation> implementation)
: TargetValue(implementation)
{

}

} // namespace generic
} // namespace machine
} // namespace lucius










