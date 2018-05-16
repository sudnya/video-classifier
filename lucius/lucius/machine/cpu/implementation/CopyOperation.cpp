/*! \file  CopyOperation.cpp
    \date  December 24, 2017
    \autor Gregory Diamos <solusstultus@gmail.com>
    \brief The source file for the CopyOperation class.
*/

// Lucius Includes
#include <lucius/machine/cpu/interface/CopyOperation.h>

#include <lucius/machine/generic/interface/IntegerData.h>
#include <lucius/machine/generic/interface/DataAccessors.h>

#include <lucius/ir/interface/BasicBlock.h>
#include <lucius/ir/interface/Use.h>
#include <lucius/ir/interface/Value.h>
#include <lucius/ir/interface/Function.h>
#include <lucius/ir/interface/Shape.h>

#include <lucius/ir/types/interface/TensorType.h>

#include <lucius/ir/target/interface/PerformanceMetrics.h>
#include <lucius/ir/target/interface/TargetValue.h>
#include <lucius/ir/target/interface/TargetValueData.h>

#include <lucius/ir/target/implementation/TargetOperationImplementation.h>

#include <lucius/matrix/interface/Matrix.h>
#include <lucius/matrix/interface/Operator.h>
#include <lucius/matrix/interface/CopyOperations.h>

// Standard Library Includes
#include <cassert>
#include <string>

namespace lucius
{
namespace machine
{
namespace cpu
{

class CopyOperationImplementation : public ir::TargetOperationImplementation
{
public:
    virtual ir::PerformanceMetrics getPerformanceMetrics() const
    {
        auto outputType = getOperand(0).getValue().getType();
        auto tensorType = ir::type_cast<ir::TensorType>(outputType);

        size_t size = tensorType.getShape().elements();

        // size flops, one for each element
        // 3 * size memory ops, two loads and one store for each element
        // 0 network ops
        return ir::PerformanceMetrics(size, size * 3, 0.0);
    }

public:
    virtual ir::BasicBlock execute()
    {
        if(getType().isTensor())
        {
            auto out = generic::getDataAsTensor(getOperand(1));
            auto in  = generic::getDataAsTensor(getOperand(0));

            matrix::copy(out, in);
        }
        else if(getType().isInteger())
        {
            size_t in = generic::getDataAsInteger(getOperand(0));

            auto outValue = ir::value_cast<ir::TargetValue>(getOperand(1).getValue());
            auto outData  = ir::data_cast<generic::IntegerData>(outValue.getData());

            outData.setInteger(in);
        }

        return getParent();
    }

public:
    virtual std::shared_ptr<ValueImplementation> clone() const
    {
        return std::make_shared<CopyOperationImplementation>(*this);
    }

public:
    virtual std::string name() const
    {
        return "cpu-copy";
    }

public:
    virtual ir::Type getType() const
    {
        return getOutputOperand().getValue().getType();
    }

};

CopyOperation::CopyOperation()
: TargetOperation(std::make_shared<CopyOperationImplementation>())
{

}

CopyOperation::~CopyOperation()
{

}

} // namespace cpu
} // namespace machine
} // namespace lucius









