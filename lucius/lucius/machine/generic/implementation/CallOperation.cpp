/*! \file  CallOperation.cpp
    \date  December 24, 2017
    \autor Gregory Diamos <solusstultus@gmail.com>
    \brief The source file for the CallOperation class.
*/

// Lucius Includes
#include <lucius/machine/generic/interface/CallOperation.h>

#include <lucius/machine/generic/interface/DataAccessors.h>

#include <lucius/ir/interface/BasicBlock.h>
#include <lucius/ir/interface/Use.h>
#include <lucius/ir/interface/Value.h>
#include <lucius/ir/interface/Function.h>
#include <lucius/ir/interface/ExternalFunction.h>

#include <lucius/ir/target/interface/PerformanceMetrics.h>
#include <lucius/ir/target/interface/TargetValue.h>

#include <lucius/ir/target/implementation/TargetControlOperationImplementation.h>

#include <lucius/matrix/interface/Matrix.h>

#include <lucius/util/interface/ForeignFunctionInterface.h>
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

static void runExternalFunction(const ir::ExternalFunction& externalFunction,
    const CallOperation::UseList& operands)
{
    auto name = externalFunction.name();

    util::ForeignFunctionArguments arguments;

    // Tensors are passed by pointer.  This list holds the objects pointed to until after the call
    std::list<matrix::Matrix> tensorArguments;
    std::list<ir::TargetValue> targetValues;

    for(auto& operand : operands)
    {
        auto value = ir::value_cast<ir::TargetValue>(operand.getValue());

        if(externalFunction.getPassArgumentsAsTargetValues())
        {
            targetValues.push_back(value);
            arguments.push_back(util::ForeignFunctionArgument(&targetValues.back()));
            continue;
        }

        auto type = value.getType();

        if(type.isInteger())
        {
            auto data = getDataAsInteger(operand);

            arguments.push_back(util::ForeignFunctionArgument(data));
        }
        else if(type.isFloat())
        {
            auto data = getDataAsFloat(operand);

            arguments.push_back(util::ForeignFunctionArgument(data));
        }
        else if(type.isTensor())
        {
            auto data = getDataAsTensor(operand);

            tensorArguments.push_back(data);

            arguments.push_back(util::ForeignFunctionArgument(&tensorArguments.back()));
        }
        else if(type.isPointer())
        {
            auto* data = getDataAsPointer(operand);

            arguments.push_back(util::ForeignFunctionArgument(data));
        }
        else
        {
            assertM(false, "TODO: add support for foreign function calls "
                "using operands of type '" + type.toString() + "'");
        }
    }

    // TODO: support return type
    util::callForeignFunction(name, arguments);
}

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

        if(functionValue.isFunction())
        {
            auto function = ir::value_cast<ir::Function>(functionValue);

            return function.getEntryBlock();
        }

        assert(functionValue.isExternalFunction());

        auto externalFunction = ir::value_cast<ir::ExternalFunction>(functionValue);

        runExternalFunction(externalFunction, UseList(++begin(), end()));

        return getBasicBlock().getNextBasicBlock();
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

    virtual BasicBlockVector getPossibleTargets() const
    {
        return {getBasicBlock().getNextBasicBlock()};
    }

    virtual bool canFallthrough() const
    {
        return true;
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








