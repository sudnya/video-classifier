/*  \file   GetOperation.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the GetOperation class.
*/

// Lucius Includes
#include <lucius/machine/generic/interface/GetOperation.h>

#include <lucius/machine/generic/interface/DataAccessors.h>

#include <lucius/ir/interface/BasicBlock.h>
#include <lucius/ir/interface/Use.h>
#include <lucius/ir/interface/Value.h>

#include <lucius/ir/types/interface/StructureType.h>

#include <lucius/ir/target/interface/TargetValue.h>
#include <lucius/ir/target/interface/TargetValueData.h>
#include <lucius/ir/target/interface/PerformanceMetrics.h>

#include <lucius/ir/target/implementation/TargetOperationImplementation.h>

// Standard Library Includes
#include <string>

namespace lucius
{
namespace machine
{
namespace generic
{

class GetOperationImplementation : public ir::TargetOperationImplementation
{
public:
    virtual ir::PerformanceMetrics getPerformanceMetrics() const
    {
        // how many memory operations are needed for a free?
        double ops = 1.0;

        return ir::PerformanceMetrics(0.0, ops, 0.0);
    }

public:
    virtual ir::BasicBlock execute()
    {
        auto index = getDataAsInteger(getOperand(1));

        auto outputValue = ir::value_cast<ir::TargetValue>(getOutputOperand().getValue());

        copyData(outputValue.getData(), getDataAtIndex(getOperand(0), index), getType());

        return getParent();
    }

public:
    virtual std::shared_ptr<ir::ValueImplementation> clone() const
    {
        return std::make_shared<GetOperationImplementation>(*this);
    }

    virtual std::string name() const
    {
        return "get";
    }

    virtual ir::Type getType() const
    {
        auto operand = getOperand(0);

        auto value = operand.getValue();

        // TODO: support other types
        assert(value.isStructure());

        auto structureType = ir::type_cast<ir::StructureType>(value.getType());

        size_t position = getDataAsInteger(getOperand(1));

        return structureType[position];
    }

};

GetOperation::GetOperation()
: TargetOperation(std::make_shared<GetOperationImplementation>())
{

}

GetOperation::~GetOperation()
{

}

} // namespace generic
} // namespace machine
} // namespace lucius





