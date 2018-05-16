/*! \file  ZerosOperation.cpp
    \date  December 24, 2017
    \autor Gregory Diamos <solusstultus@gmail.com>
    \brief The source file for the ZerosOperation class.
*/

// Lucius Includes
#include <lucius/machine/cpu/interface/ZerosOperation.h>

#include <lucius/machine/generic/interface/DataAccessors.h>

#include <lucius/ir/interface/BasicBlock.h>
#include <lucius/ir/interface/Use.h>
#include <lucius/ir/interface/Value.h>
#include <lucius/ir/interface/Function.h>
#include <lucius/ir/interface/Shape.h>

#include <lucius/ir/types/interface/TensorType.h>

#include <lucius/ir/target/interface/PerformanceMetrics.h>

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

class ZerosOperationImplementation : public ir::TargetOperationImplementation
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
        auto out = generic::getDataAsTensor(getOperand(0));

        matrix::zeros(out);

        return getParent();
    }

public:
    virtual std::shared_ptr<ValueImplementation> clone() const
    {
        return std::make_shared<ZerosOperationImplementation>(*this);
    }

public:
    virtual std::string name() const
    {
        return "cpu-zeros";
    }

public:
    virtual ir::Type getType() const
    {
        return getOutputOperand().getValue().getType();
    }

};

ZerosOperation::ZerosOperation()
: TargetOperation(std::make_shared<ZerosOperationImplementation>())
{

}

ZerosOperation::~ZerosOperation()
{

}

} // namespace cpu
} // namespace machine
} // namespace lucius








