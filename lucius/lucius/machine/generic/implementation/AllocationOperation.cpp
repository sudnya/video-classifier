/*  \file   AllocationOperation.cpp
    \author Gregory Diamos
    \date   October 9, 2017
    \brief  The source file for the AllocationOperation class.
*/

// Lucius Includes
#include <lucius/machine/generic/interface/AllocationOperation.h>

#include <lucius/ir/interface/BasicBlock.h>
#include <lucius/ir/interface/Use.h>
#include <lucius/ir/interface/Value.h>

#include <lucius/ir/target/interface/TargetValue.h>
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

class AllocationOperationImplementation : public ir::TargetOperationImplementation
{
public:
    virtual ir::PerformanceMetrics getPerformanceMetrics() const
    {
        // how many memory operations are needed for a malloc?
        double ops = 1.0;

        return ir::PerformanceMetrics(0.0, ops, 0.0);
    }

public:
    virtual ir::BasicBlock execute()
    {
        auto operand = getOutputOperand();

        auto value = ir::value_cast<ir::TargetValue>(operand.getValue());

        value.allocateData();

        return getParent();
    }

public:
    virtual std::shared_ptr<ir::ValueImplementation> clone() const
    {
        return std::make_shared<AllocationOperationImplementation>(*this);
    }

    virtual std::string name() const
    {
        return "require-allocated";
    }

    virtual std::string toString() const
    {
        return name() + " " + getOutputOperand().toString();
    }

    virtual ir::Type getType() const
    {
        auto operand = getOutputOperand();

        auto value = ir::value_cast<ir::TargetValue>(operand.getValue());

        return value.getType();
    }

};

AllocationOperation::AllocationOperation()
: ir::TargetOperation(std::make_shared<AllocationOperationImplementation>())
{

}

AllocationOperation::~AllocationOperation()
{
    // intentionally blank
}

} // namespace generic
} // namespace machine
} // namespace lucius



