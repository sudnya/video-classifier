/*  \file   GradientOperation.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the GradientOperation class.
*/

// Lucius Includes
#include <lucius/machine/generic/interface/GradientOperation.h>

#include <lucius/ir/interface/BasicBlock.h>
#include <lucius/ir/interface/Use.h>
#include <lucius/ir/interface/Value.h>

#include <lucius/ir/target/interface/TargetValue.h>
#include <lucius/ir/target/interface/PerformanceMetrics.h>

#include <lucius/ir/target/implementation/TargetComputeGradientOperationImplementation.h>

#include <lucius/util/interface/debug.h>

// Standard Library Includes
#include <string>

namespace lucius
{
namespace machine
{
namespace generic
{

class GradientOperationImplementation : public ir::TargetComputeGradientOperationImplementation
{
public:
    virtual ir::PerformanceMetrics getPerformanceMetrics() const
    {
        // Gradient operations should never be executed
        return ir::PerformanceMetrics(0.0, 0.0, 0.0);
    }

public:
    virtual ir::BasicBlock execute()
    {
        assertM(false, "Gradient operations should be compiled out.");

        return getParent();
    }

public:
    virtual std::shared_ptr<ir::ValueImplementation> clone() const
    {
        return std::make_shared<GradientOperationImplementation>(*this);
    }

    virtual std::string name() const
    {
        return "generic-gradient";
    }

    virtual ir::Type getType() const
    {
        auto operand = getOperand(0);

        auto value = ir::value_cast<ir::TargetValue>(operand.getValue());

        return value.getType();
    }

};

GradientOperation::GradientOperation()
: TargetOperation(std::make_shared<GradientOperationImplementation>())
{

}

GradientOperation::~GradientOperation()
{

}

} // namespace generic
} // namespace machine
} // namespace lucius





