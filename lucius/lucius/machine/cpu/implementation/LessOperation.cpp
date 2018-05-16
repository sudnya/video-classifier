/*! \file  LessOperation.cpp
    \date  December 24, 2017
    \autor Gregory Diamos <solusstultus@gmail.com>
    \brief The source file for the LessOperation class.
*/

// Lucius Includes
#include <lucius/machine/cpu/interface/LessOperation.h>

#include <lucius/machine/generic/interface/IntegerData.h>
#include <lucius/machine/generic/interface/DataAccessors.h>

#include <lucius/ir/interface/BasicBlock.h>
#include <lucius/ir/interface/Use.h>
#include <lucius/ir/interface/Value.h>

#include <lucius/ir/target/interface/TargetValueData.h>
#include <lucius/ir/target/interface/TargetValue.h>

#include <lucius/ir/target/interface/PerformanceMetrics.h>

#include <lucius/ir/target/implementation/TargetOperationImplementation.h>

// Standard Library Includes
#include <cassert>
#include <string>

namespace lucius
{
namespace machine
{
namespace cpu
{

class LessOperationImplementation : public ir::TargetOperationImplementation
{
public:
    virtual ir::PerformanceMetrics getPerformanceMetrics() const
    {
        // one flop for the comparison
        // two loads for the elements being compared and one store for the result
        // 0 network ops
        return ir::PerformanceMetrics(1.0, 3.0, 0.0);
    }

public:
    virtual ir::BasicBlock execute()
    {
        assert(getType().isInteger());

        auto left  = generic::getDataAsInteger(getOperand(0));
        auto right = generic::getDataAsInteger(getOperand(1));

        auto outValue = ir::value_cast<ir::TargetValue>(getOutputOperand().getValue());
        auto out = ir::data_cast<generic::IntegerData>(outValue.getData());

        out.setInteger(left < right ? 1 : 0);

        return getParent();
    }

public:
    virtual std::shared_ptr<ValueImplementation> clone() const
    {
        return std::make_shared<LessOperationImplementation>(*this);
    }

public:
    virtual std::string name() const
    {
        return "cpu-less";
    }

public:
    virtual ir::Type getType() const
    {
        assert(getOperands().size() == 3);
        return getOutputOperand().getValue().getType();
    }

};

LessOperation::LessOperation()
: TargetOperation(std::make_shared<LessOperationImplementation>())
{

}

LessOperation::~LessOperation()
{

}

} // namespace cpu
} // namespace machine
} // namespace lucius








