/*! \file  ReduceOperation.cpp
    \date  December 24, 2017
    \autor Gregory Diamos <solusstultus@gmail.com>
    \brief The source file for the ReduceOperation class.
*/

// Lucius Includes
#include <lucius/machine/cpu/interface/ReduceOperation.h>

#include <lucius/machine/generic/interface/IntegerData.h>
#include <lucius/machine/generic/interface/DataAccessors.h>

#include <lucius/ir/interface/BasicBlock.h>
#include <lucius/ir/interface/Use.h>
#include <lucius/ir/interface/Value.h>
#include <lucius/ir/interface/Function.h>
#include <lucius/ir/interface/Shape.h>
#include <lucius/ir/interface/ShapeUtilities.h>

#include <lucius/ir/types/interface/TensorType.h>
#include <lucius/ir/types/interface/ShapeType.h>

#include <lucius/ir/target/interface/PerformanceMetrics.h>
#include <lucius/ir/target/interface/TargetValue.h>
#include <lucius/ir/target/interface/TargetValueData.h>

#include <lucius/ir/target/implementation/TargetOperationImplementation.h>

#include <lucius/matrix/interface/Matrix.h>
#include <lucius/matrix/interface/Operator.h>

#include <lucius/matrix/interface/MatrixOperations.h>

// Standard Library Includes
#include <cassert>
#include <string>

namespace lucius
{
namespace machine
{
namespace cpu
{

class ReduceOperationImplementation : public ir::TargetOperationImplementation
{
public:
    virtual ir::PerformanceMetrics getPerformanceMetrics() const
    {
        auto outputTensorType = ir::type_cast<ir::TensorType>(getType());

        auto valueSize = outputTensorType.getPrecision().size();

        auto inputType = getOperand(0).getValue().getType();
        auto inputTensorType = ir::type_cast<ir::TensorType>(inputType);

        size_t inputSize = inputTensorType.getShape().elements();
        size_t outputSize = outputTensorType.getShape().elements();

        // size flops, two for each input element
        // one load for each input, one store for each output
        // 0 network ops
        return ir::PerformanceMetrics(inputSize, (inputSize + outputSize) * valueSize, 0.0);
    }

public:
    virtual ir::BasicBlock execute()
    {
        assert(getType().isTensor());

        auto input = generic::getDataAsTensor(getOperand(0));
        auto shape = generic::getDataAsDimension(getOperand(1));

        auto applyOperator = generic::getDataAsOperator(getOperand(2));

        auto out = generic::getDataAsTensor(getOperand(3));

        matrix::reduce(out, input, shape, applyOperator.getStaticOperator());

        return getParent();
    }

public:
    virtual std::shared_ptr<ValueImplementation> clone() const
    {
        return std::make_shared<ReduceOperationImplementation>(*this);
    }

public:
    virtual std::string name() const
    {
        return "cpu-reduce";
    }

public:
    virtual ir::Type getType() const
    {
        assert(getOperands().size() == 4);

        // input tensor shape
        auto inputType = getOperand(0).getValue().getType();
        auto inputTensorType = ir::type_cast<ir::TensorType>(inputType);

        auto inputShape = inputTensorType.getShape();

        auto reduceType = getOperand(1).getValue().getType();
        auto reduceShapeType = ir::type_cast<ir::ShapeType>(reduceType);

        auto reduceShape = reduceShapeType.getShape();

        ir::Shape outputShape({1});

        if(!reduceShape.empty())
        {
            outputShape = removeElements(inputShape, reduceShape);
        }

        return ir::TensorType(outputShape, inputTensorType.getPrecision());
    }

};

ReduceOperation::ReduceOperation()
: TargetOperation(std::make_shared<ReduceOperationImplementation>())
{

}

ReduceOperation::~ReduceOperation()
{

}

} // namespace cpu
} // namespace machine
} // namespace lucius








