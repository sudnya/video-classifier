/*  \file   StructureValue.cpp
    \author Gregory Diamos
    \date   October 11, 2017
    \brief  The source file for the StructureValue class.
*/

// Lucius Includes
#include <lucius/machine/generic/interface/StructureValue.h>

#include <lucius/machine/generic/interface/StructureData.h>
#include <lucius/machine/generic/interface/DataFactory.h>

#include <lucius/ir/types/interface/StructureType.h>

#include <lucius/ir/interface/Use.h>

#include <lucius/ir/target/implementation/TargetValueImplementation.h>

// Standard Library Includes
#include <string>

namespace lucius
{
namespace machine
{
namespace generic
{

class StructureValueImplementation : public ir::TargetValueImplementation
{
public:
    using Type = ir::Type;
    using Context = ir::Context;
    using TargetValueData = ir::TargetValueData;

public:
    StructureValueImplementation(const Type& type, Context& context)
    : _type(ir::type_cast<ir::StructureType>(type))
    {
        bindToContext(&context);
    }

public:
    virtual std::shared_ptr<ValueImplementation> clone() const
    {
        return std::make_shared<StructureValueImplementation>(*this);
    }

    std::string toString() const
    {
        return "GenericStructure(" + getType().toString() + ")" + "%" + std::to_string(getId());
    }

public:
    Type getType() const
    {
        return _type;
    }

public:
    void allocateData()
    {
        setData(DataFactory::create(_type));
    }

    void freeData()
    {
        setData(TargetValueData());
    }

private:
    ir::StructureType _type;
};

StructureValue::StructureValue(const ir::Type& type, ir::Context& context)
: StructureValue(std::make_shared<StructureValueImplementation>(type, context))
{

}

StructureValue::StructureValue(std::shared_ptr<ir::ValueImplementation> implementation)
: TargetValue(implementation)
{

}

} // namespace generic
} // namespace machine
} // namespace lucius










