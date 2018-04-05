/*  \file   AllocationOperation.cpp
    \author Gregory Diamos
    \date   October 9, 2017
    \brief  The source file for the AllocationOperation class.
*/

// Lucius Includes
#include <lucius/ir/target/interface/AllocationOperation.h>

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

namespace ir
{

class AllocationOperationImplementation : public TargetOperationImplementation
{
public:
    virtual PerformanceMetrics getPerformanceMetrics() const
    {
        // how many memory operations are needed for a malloc?
        double ops = 1.0;

        return PerformanceMetrics(0.0, ops, 0.0);
    }

public:
    virtual BasicBlock execute()
    {
        auto operand = getOutputOperand();

        auto value = ir::value_cast<TargetValue>(operand.getValue());

        value.allocateData();

        return getParent();
    }

public:
    virtual std::shared_ptr<ValueImplementation> clone() const
    {
        return std::make_shared<AllocationOperationImplementation>(*this);
    }

    virtual std::string name() const
    {
        return "allocate";
    }

    virtual Type getType() const
    {
        auto operand = getOutputOperand();

        auto value = ir::value_cast<TargetValue>(operand.getValue());

        return value.getType();
    }

};

AllocationOperation::AllocationOperation(TargetValue allocatedValue)
: TargetOperation(std::make_shared<AllocationOperationImplementation>())
{
    setOutputOperand(allocatedValue);
}

AllocationOperation::~AllocationOperation()
{
    // intentionally blank
}

} // namespace ir
} // namespace lucius



