/*! \file  CallOperation.cpp
    \date  December 24, 2017
    \autor Gregory Diamos <solusstultus@gmail.com>
    \brief The source file for the CallOperation class.
*/

// Lucius Includes
#include <lucius/machine/generic/interface/CallOperation.h>

#include <lucius/ir/interface/BasicBlock.h>
#include <lucius/ir/interface/Use.h>
#include <lucius/ir/interface/Value.h>
#include <lucius/ir/interface/Function.h>

#include <lucius/ir/target/interface/PerformanceMetrics.h>

#include <lucius/ir/target/implementation/TargetControlOperationImplementation.h>

// Standard Library Includes
#include <cassert>
#include <string>

namespace lucius
{
namespace machine
{
namespace generic
{

class CallOperationImplementation : public ir::TargetControlOperationImplementation
{
public:
    /*! \brief Get the performance metrics for this operations. */
    virtual ir::PerformanceMetrics getPerformanceMetrics() const
    {
        // 1 flops for the branch
        // 1 memory op for the target
        // 0 network ops
        return ir::PerformanceMetrics(1.0, 1.0, 0.0);
    }

public:
    /*! \brief Execute the operation. */
    virtual ir::BasicBlock execute()
    {
        auto& functionOperand = getOperand(0);
        auto functionValue = functionOperand.getValue();

        assert(functionValue.isFunction());

        auto function = ir::value_cast<ir::Function>(functionValue);

        return function.getEntryBlock();
    }

public:
    virtual std::shared_ptr<ValueImplementation> clone() const
    {
        return std::make_shared<CallOperationImplementation>(*this);
    }

public:
    virtual std::string name() const
    {
        return "generic-call";
    }

    virtual bool isCall() const
    {
        return true;
    }

public:
    virtual ir::Type getType() const
    {
        assert(!getOperands().empty());
        return getOperand(0).getValue().getType();
    }

};

CallOperation::CallOperation()
: TargetOperation(std::make_shared<CallOperationImplementation>())
{

}

CallOperation::~CallOperation()
{

}

} // namespace generic
} // namespace machine
} // namespace lucius








