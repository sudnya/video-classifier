/*  \file   FreeOperation.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the FreeOperation class.
*/

// Lucius Includes
#include <lucius/machine/generic/interface/FreeOperation.h>

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

class FreeOperationImplementation : public ir::TargetOperationImplementation
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
        auto operand = getOperand(0);

        auto value = ir::value_cast<ir::TargetValue>(operand.getValue());

        value.freeData();

        return getParent();
    }

public:
    virtual std::shared_ptr<ir::ValueImplementation> clone() const
    {
        return std::make_shared<FreeOperationImplementation>(*this);
    }

    virtual std::string name() const
    {
        return "require-freed";
    }

    virtual ir::Type getType() const
    {
        auto operand = getOperand(0);

        auto value = ir::value_cast<ir::TargetValue>(operand.getValue());

        return value.getType();
    }

};

FreeOperation::FreeOperation()
: TargetOperation(std::make_shared<FreeOperationImplementation>())
{

}

FreeOperation::~FreeOperation()
{

}

} // namespace generic
} // namespace machine
} // namespace lucius




