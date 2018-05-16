/*! \file  PHIOperation.cpp
    \date  December 24, 2017
    \autor Gregory Diamos <solusstultus@gmail.com>
    \brief The source file for the PHIOperation class.
*/

// Lucius Includes
#include <lucius/machine/generic/interface/PHIOperation.h>

#include <lucius/ir/interface/BasicBlock.h>
#include <lucius/ir/interface/Use.h>
#include <lucius/ir/interface/Value.h>
#include <lucius/ir/interface/Function.h>
#include <lucius/ir/interface/ExternalFunction.h>

#include <lucius/ir/target/interface/PerformanceMetrics.h>
#include <lucius/ir/target/interface/TargetValue.h>

#include <lucius/ir/target/implementation/TargetOperationImplementation.h>

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

class PHIOperationImplementation : public ir::TargetOperationImplementation
{
public:
    PHIOperationImplementation()
    {
        // intentionally blank
    }

    PHIOperationImplementation(const ir::Type& type)
    : _type(type)
    {

    }

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
        assertM(false, "PHI operations should have been removed prior to execution.");

        return getBasicBlock();
    }

public:
    virtual std::shared_ptr<ValueImplementation> clone() const
    {
        return std::make_shared<PHIOperationImplementation>(*this);
    }

public:
    virtual std::string name() const
    {
        return "generic-phi";
    }

public:
    virtual ir::Type getType() const
    {
        return _type;
    }

    using BasicBlockVector = PHIOperation::BasicBlockVector;

    virtual BasicBlockVector getPossibleTargets() const
    {
        return _incomingBlocks;
    }

    virtual bool canFallthrough() const
    {
        return true;
    }

    virtual bool isPHI() const
    {
        return true;
    }

public:

    const BasicBlockVector& getIncomingBasicBlocks() const
    {
        return _incomingBlocks;
    }

    void addIncomingValue(const ir::TargetValue& v, const ir::BasicBlock& incoming)
    {
        appendOperand(v);
        addIncomingBasicBlock(incoming);
    }

    void addIncomingBasicBlock(const ir::BasicBlock& incoming)
    {
        _incomingBlocks.push_back(incoming);
    }

    void setType(const ir::Type& type)
    {
        _type = type;
    }

private:
    BasicBlockVector _incomingBlocks;

private:
    ir::Type _type;

};

PHIOperation::PHIOperation()
: TargetOperation(std::make_shared<PHIOperationImplementation>())
{

}

PHIOperation::PHIOperation(const ir::Type& type)
: TargetOperation(std::make_shared<PHIOperationImplementation>(type))
{

}

PHIOperation::PHIOperation(std::shared_ptr<ir::ValueImplementation> implementation)
: TargetOperation(implementation)
{

}

PHIOperation::~PHIOperation()
{

}

const PHIOperation::BasicBlockVector& PHIOperation::getIncomingBasicBlocks() const
{
    return getPHIImplementation()->getIncomingBasicBlocks();
}

void PHIOperation::addIncomingValue(const ir::TargetValue& v, const ir::BasicBlock& incoming)
{
    getPHIImplementation()->addIncomingValue(v, incoming);
}

void PHIOperation::addIncomingBasicBlock(const ir::BasicBlock& incoming)
{
    getPHIImplementation()->addIncomingBasicBlock(incoming);
}

void PHIOperation::setType(const ir::Type& type)
{
    getPHIImplementation()->setType(type);
}

std::shared_ptr<PHIOperationImplementation> PHIOperation::getPHIImplementation() const
{
    return std::static_pointer_cast<PHIOperationImplementation>(getValueImplementation());
}

} // namespace generic
} // namespace machine
} // namespace lucius










