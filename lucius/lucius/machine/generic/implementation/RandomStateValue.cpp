/*  \file   RandomStateValue.cpp
    \author Gregory Diamos
    \date   October 11, 2017
    \brief  The source file for the RandomStateValue class.
*/

// Lucius Includes
#include <lucius/machine/generic/interface/RandomStateValue.h>

#include <lucius/ir/interface/Shape.h>
#include <lucius/ir/interface/Use.h>

#include <lucius/machine/generic/interface/RandomStateData.h>

#include <lucius/ir/types/interface/RandomStateType.h>

#include <lucius/ir/target/implementation/TargetValueImplementation.h>

namespace lucius
{
namespace machine
{
namespace generic
{

class RandomStateValueImplementation : public ir::TargetValueImplementation
{
public:
    using RandomStateType = ir::RandomStateType;
    using TargetValueData = ir::TargetValueData;
    using Context         = ir::Context;
    using Type            = ir::Type;

public:
    RandomStateValueImplementation(Context& context)
    {
        bindToContext(&context);
    }

public:
    virtual std::shared_ptr<ValueImplementation> clone() const
    {
        return std::make_shared<RandomStateValueImplementation>(*this);
    }

    std::string toString() const
    {
        return "GenericRandomState(" + getType().toString() + ")" + "%" + std::to_string(getId());
    }

public:
    Type getType() const
    {
        return RandomStateType();
    }

public:
    void allocateData()
    {
        setData(RandomStateData());
    }

    void freeData()
    {
        setData(TargetValueData());
    }
};

RandomStateValue::RandomStateValue(Context& context)
: RandomStateValue(std::make_shared<RandomStateValueImplementation>(context))
{

}

RandomStateValue::RandomStateValue(std::shared_ptr<ir::ValueImplementation> implementation)
: TargetValue(implementation)
{

}

} // namespace generic
} // namespace machine
} // namespace lucius










