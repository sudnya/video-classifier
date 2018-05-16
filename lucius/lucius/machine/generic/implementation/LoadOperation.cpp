/*  \file   LoadOperation.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the LoadOperation class.
*/

// Lucius Includes
#include <lucius/machine/generic/interface/LoadOperation.h>

#include <lucius/machine/generic/interface/DataAccessors.h>

#include <lucius/ir/interface/Shape.h>
#include <lucius/ir/interface/BasicBlock.h>
#include <lucius/ir/interface/Use.h>
#include <lucius/ir/interface/Value.h>

#include <lucius/ir/types/interface/TensorType.h>

#include <lucius/ir/target/interface/TargetValue.h>
#include <lucius/ir/target/interface/TargetValueData.h>
#include <lucius/ir/target/interface/PerformanceMetrics.h>

#include <lucius/ir/target/implementation/TargetOperationImplementation.h>

#include <lucius/matrix/interface/Matrix.h>
#include <lucius/matrix/interface/CopyOperations.h>

// Standard Library Includes
#include <string>

namespace lucius
{
namespace machine
{
namespace generic
{

class LoadOperationImplementation : public ir::TargetOperationImplementation
{
public:
    virtual ir::PerformanceMetrics getPerformanceMetrics() const
    {
        auto outputType = getOutputOperand().getValue().getType();
        auto tensorType = ir::type_cast<ir::TensorType>(outputType);

        size_t size = tensorType.getShape().elements();

        // loads are like copies from vairables into values (1 load, 1 store)
        double memoryOps = 2.0 * size;

        return ir::PerformanceMetrics(0.0, memoryOps, 0.0);
    }

public:
    virtual ir::BasicBlock execute()
    {
        auto out = generic::getDataAsTensor(getOutputOperand());
        auto in  = generic::getDataAsTensor(getOperand(0));

        matrix::copy(out, in);

        return getParent();
    }

public:
    virtual std::shared_ptr<ir::ValueImplementation> clone() const
    {
        return std::make_shared<LoadOperationImplementation>(*this);
    }

    virtual std::string name() const
    {
        return "load";
    }

    virtual ir::Type getType() const
    {
        return getOperand(0).getValue().getType();
    }

};

LoadOperation::LoadOperation()
: TargetOperation(std::make_shared<LoadOperationImplementation>())
{

}

LoadOperation::~LoadOperation()
{

}

} // namespace generic
} // namespace machine
} // namespace lucius






