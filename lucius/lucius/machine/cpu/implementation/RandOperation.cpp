/*! \file  RandOperation.cpp
    \date  December 24, 2017
    \autor Gregory Diamos <solusstultus@gmail.com>
    \brief The source file for the RandOperation class.
*/

// Lucius Includes
#include <lucius/machine/cpu/interface/RandOperation.h>

#include <lucius/machine/generic/interface/RandomStateData.h>
#include <lucius/machine/generic/interface/StructureData.h>
#include <lucius/machine/generic/interface/TensorData.h>

#include <lucius/ir/interface/BasicBlock.h>
#include <lucius/ir/interface/Use.h>
#include <lucius/ir/interface/Value.h>
#include <lucius/ir/interface/Shape.h>

#include <lucius/ir/types/interface/TensorType.h>
#include <lucius/ir/types/interface/StructureType.h>

#include <lucius/ir/target/interface/PerformanceMetrics.h>
#include <lucius/ir/target/interface/TargetValue.h>
#include <lucius/ir/target/interface/TargetValueData.h>

#include <lucius/ir/target/implementation/TargetOperationImplementation.h>

#include <lucius/matrix/interface/Matrix.h>
#include <lucius/matrix/interface/RandomOperations.h>

// Standard Library Includes
#include <cassert>
#include <string>

namespace lucius
{
namespace machine
{
namespace cpu
{

class RandOperationImplementation : public ir::TargetOperationImplementation
{
public:
    virtual ir::PerformanceMetrics getPerformanceMetrics() const
    {
        auto outputType    = getOperand(1).getValue().getType();
        auto structureType = ir::type_cast<ir::StructureType>(outputType);
        auto tensorType    = ir::type_cast<ir::TensorType>(structureType[0]);

        size_t size = tensorType.getShape().elements();

        // size flops, one for each element
        // one store for each element
        // 0 network ops
        return ir::PerformanceMetrics(size, size, 0.0);
    }

public:
    virtual ir::BasicBlock execute()
    {
        auto outputValue = ir::value_cast<ir::TargetValue>(getOutputOperand().getValue());
        auto outputData  = ir::data_cast<generic::StructureData>(outputValue.getData());

        auto state = ir::data_cast<generic::RandomStateData>(getOperandData(0));

        auto outputTensor = ir::data_cast<generic::TensorData>(outputData[0]);
        auto outputState  = ir::data_cast<generic::RandomStateData>(outputData[1]);

        auto outputMatrix = outputTensor.getTensor();

        outputState.getRandomState() = state.getRandomState();

        matrix::swapDefaultRandomState(outputState.getRandomState());
        matrix::rand(outputMatrix);
        matrix::swapDefaultRandomState(outputState.getRandomState());

        return getParent();
    }

public:
    virtual std::shared_ptr<ValueImplementation> clone() const
    {
        return std::make_shared<RandOperationImplementation>(*this);
    }

public:
    virtual std::string name() const
    {
        return "cpu-rand";
    }

public:
    virtual ir::Type getType() const
    {
        return getOutputOperand().getValue().getType();
    }

};

RandOperation::RandOperation()
: TargetOperation(std::make_shared<RandOperationImplementation>())
{

}

RandOperation::~RandOperation()
{

}

} // namespace cpu
} // namespace machine
} // namespace lucius










