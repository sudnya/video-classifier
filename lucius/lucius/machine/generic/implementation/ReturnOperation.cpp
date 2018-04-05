/*! \file  ReturnOperation.cpp
    \date  December 24, 2017
    \autor Gregory Diamos <solusstultus@gmail.com>
    \brief The source file for the ReturnOperation class.
*/

// Lucius Includes
#include <lucius/machine/generic/interface/ReturnOperation.h>

#include <lucius/ir/interface/BasicBlock.h>
#include <lucius/ir/interface/Use.h>
#include <lucius/ir/interface/Value.h>
#include <lucius/ir/interface/Function.h>

#include <lucius/ir/types/interface/VoidType.h>

#include <lucius/ir/target/interface/PerformanceMetrics.h>

#include <lucius/ir/target/implementation/TargetControlOperationImplementation.h>

#include <lucius/util/interface/debug.h>

// Standard Library Includes
#include <cassert>
#include <string>

namespace lucius
{
namespace machine
{
namespace generic
{

class ReturnOperationImplementation : public ir::TargetControlOperationImplementation
{
public:
    /*! \brief Get the performance metrics for this operations. */
    virtual ir::PerformanceMetrics getPerformanceMetrics() const
    {
        // 1 flops for the branch
        // 1 memory op for the return operand
        // 0 network ops
        return ir::PerformanceMetrics(1.0, 1.0, 0.0);
    }

public:
    /*! \brief Execute the operation. */
    virtual ir::BasicBlock execute()
    {
        return getParent();
    }

public:
    virtual std::shared_ptr<ValueImplementation> clone() const
    {
        return std::make_shared<ReturnOperationImplementation>(*this);
    }

public:
    virtual std::string name() const
    {
        return "generic-return";
    }

    virtual bool isReturn() const
    {
        return true;
    }

public:
    virtual ir::Type getType() const
    {
        if(getOperands().empty())
        {
            return ir::VoidType();
        }

        return getOperand(0).getValue().getType();
    }

    virtual std::string toString() const
    {
        std::stringstream stream;

        stream << name() << " ";

        auto operandIterator = getOperands().begin();

        if(operandIterator != getOperands().end())
        {
            auto& operand = *operandIterator;

            stream << operand.getValue().toSummaryString();

            ++operandIterator;
        }

        for( ; operandIterator != getOperands().end(); ++operandIterator)
        {
            auto& operand = *operandIterator;

            stream << ", " << operand.getValue().toSummaryString();
        }

        return stream.str();
    }

    virtual BasicBlockVector getPossibleTargets() const
    {
        return {};
    }

};

ReturnOperation::ReturnOperation()
: ir::TargetOperation(std::make_shared<ReturnOperationImplementation>())
{

}

ReturnOperation::~ReturnOperation()
{

}

} // namespace generic
} // namespace machine
} // namespace lucius







