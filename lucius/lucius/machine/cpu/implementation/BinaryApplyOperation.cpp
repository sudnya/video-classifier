/*! \file  BinaryApplyOperation.cpp
    \date  December 24, 2017
    \autor Gregory Diamos <solusstultus@gmail.com>
    \brief The source file for the BinaryApplyOperation class.
*/

// Lucius Includes
#include <lucius/machine/cpu/interface/BinaryApplyOperation.h>

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

#include <lucius/matrix/interface/Scalar.h>
#include <lucius/matrix/interface/Matrix.h>
#include <lucius/matrix/interface/Operator.h>

#include <lucius/matrix/interface/MatrixOperations.h>
#include <lucius/matrix/interface/ScalarOperations.h>

// Standard Library Includes
#include <cassert>
#include <string>

namespace lucius
{
namespace machine
{
namespace cpu
{

class BinaryApplyOperationImplementation : public ir::TargetOperationImplementation
{
public:
    virtual ir::PerformanceMetrics getPerformanceMetrics() const
    {
        auto outputType = getOutputOperand().getValue().getType();
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
            auto left  = generic::getDataAsTensor(getOperand(0));
            auto right = generic::getDataAsTensor(getOperand(1));

            auto applyOperator = generic::getDataAsOperator(getOperand(2));

            auto out = generic::getDataAsTensor(getOperand(3));

            matrix::apply(out, left, right, applyOperator.getStaticOperator());
        }
        else if(getType().isInteger())
        {
            auto left  = matrix::Scalar(generic::getDataAsInteger(getOperand(0)));
            auto right = matrix::Scalar(generic::getDataAsInteger(getOperand(1)));

            auto applyOperator = generic::getDataAsOperator(getOperand(2));

            auto outValue = ir::value_cast<ir::TargetValue>(getOperand(3).getValue());
            auto outData  = ir::data_cast<generic::IntegerData>(outValue.getData());

            matrix::Scalar result;

            matrix::apply(result, left, right, applyOperator.getStaticOperator());

            outData.setInteger(result.get<size_t>());
        }

        return getParent();
    }

public:
    virtual std::shared_ptr<ValueImplementation> clone() const
    {
        return std::make_shared<BinaryApplyOperationImplementation>(*this);
    }

public:
    virtual std::string name() const
    {
        return "cpu-binary-apply";
    }

public:
    virtual ir::Type getType() const
    {
        assert(getOperands().size() == 4);
        return getOutputOperand().getValue().getType();
    }

};

BinaryApplyOperation::BinaryApplyOperation()
: TargetOperation(std::make_shared<BinaryApplyOperationImplementation>())
{

}

BinaryApplyOperation::~BinaryApplyOperation()
{

}

} // namespace cpu
} // namespace machine
} // namespace lucius







