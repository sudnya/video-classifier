/*! \file  BranchOperation.cpp
    \date  December 24, 2017
    \autor Gregory Diamos <solusstultus@gmail.com>
    \brief The source file for the BranchOperation class.
*/

// Lucius Includes
#include <lucius/machine/generic/interface/BranchOperation.h>

#include <lucius/ir/interface/BasicBlock.h>
#include <lucius/ir/interface/Use.h>
#include <lucius/ir/interface/Value.h>
#include <lucius/ir/interface/Function.h>
#include <lucius/ir/interface/ExternalFunction.h>

#include <lucius/ir/target/interface/PerformanceMetrics.h>
#include <lucius/ir/target/interface/TargetValue.h>

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

class BranchOperationImplementation : public ir::TargetControlOperationImplementation
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
    /*! \brief Execute the branch, using the condition to select the next block. */
    virtual ir::BasicBlock execute()
    {
        auto condition = getOperandDataAsInteger(0);

        auto target      = ir::value_cast<ir::BasicBlock>(getOperand(1).getValue());
        auto fallthrough = ir::value_cast<ir::BasicBlock>(getOperand(2).getValue());

        if(condition != 0)
        {
            return target;
        }
        else
        {
            return fallthrough;
        }
    }

public:
    virtual std::shared_ptr<ValueImplementation> clone() const
    {
        return std::make_shared<BranchOperationImplementation>(*this);
    }

public:
    virtual std::string name() const
    {
        return "generic-branch";
    }

public:
    virtual ir::Type getType() const
    {
        return ir::Type(ir::Type::BasicBlockId);
    }

    virtual BasicBlockVector getPossibleTargets() const
    {
        auto target      = ir::value_cast<ir::BasicBlock>(getOperand(1).getValue());
        auto fallthrough = ir::value_cast<ir::BasicBlock>(getOperand(2).getValue());

        return {target, fallthrough};
    }

    virtual bool canFallthrough() const
    {
        return false;
    }

};

BranchOperation::BranchOperation()
: TargetOperation(std::make_shared<BranchOperationImplementation>())
{

}

BranchOperation::~BranchOperation()
{

}

} // namespace generic
} // namespace machine
} // namespace lucius









