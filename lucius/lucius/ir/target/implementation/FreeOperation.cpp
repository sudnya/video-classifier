/*  \file   FreeOperation.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the FreeOperation class.
*/

// Lucius Includes
#include <lucius/ir/target/interface/FreeOperation.h>

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

class FreeOperationImplementation : public TargetOperationImplementation
{
public:
    virtual PerformanceMetrics getPerformanceMetrics() const
    {
        // how many memory operations are needed for a free?
        double ops = 1.0;

        return PerformanceMetrics(0.0, ops, 0.0);
    }

public:
    virtual BasicBlock execute()
    {
        auto operand = getOperand(0);

        auto value = ir::value_cast<TargetValue>(operand.getValue());

        value.freeData();

        return getParent();
    }

public:
    virtual std::shared_ptr<ValueImplementation> clone() const
    {
        return std::make_shared<FreeOperationImplementation>(*this);
    }

    virtual std::string name() const
    {
        return "free";
    }

    virtual Type getType() const
    {
        auto operand = getOperand(0);

        auto value = ir::value_cast<TargetValue>(operand.getValue());

        return value.getType();
    }

};

FreeOperation::FreeOperation(TargetValue allocatedValue)
: TargetOperation(std::make_shared<FreeOperationImplementation>())
{
    appendOperand(allocatedValue);
}

FreeOperation::~FreeOperation()
{

}

} // namespace ir
} // namespace lucius




