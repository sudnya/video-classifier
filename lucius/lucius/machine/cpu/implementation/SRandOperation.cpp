/*! \file  SRandOperation.cpp
    \date  December 24, 2017
    \autor Gregory Diamos <solusstultus@gmail.com>
    \brief The source file for the SRandOperation class.
*/

// Lucius Includes
#include <lucius/machine/cpu/interface/SRandOperation.h>

#include <lucius/machine/generic/interface/RandomStateData.h>
#include <lucius/machine/generic/interface/DataAccessors.h>

#include <lucius/ir/interface/BasicBlock.h>
#include <lucius/ir/interface/Use.h>
#include <lucius/ir/interface/Value.h>
#include <lucius/ir/interface/Function.h>
#include <lucius/ir/interface/Shape.h>

#include <lucius/ir/types/interface/RandomStateType.h>

#include <lucius/ir/target/interface/PerformanceMetrics.h>
#include <lucius/ir/target/interface/TargetValue.h>
#include <lucius/ir/target/interface/TargetValueData.h>

#include <lucius/ir/target/implementation/TargetOperationImplementation.h>

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

class SRandOperationImplementation : public ir::TargetOperationImplementation
{
public:
    virtual ir::PerformanceMetrics getPerformanceMetrics() const
    {
        // size flops, one for each element
        // not sure here, 1 flop, 1 load/store for the state
        // 0 network ops
        return ir::PerformanceMetrics(1.0, 2.0, 0.0);
    }

public:
    virtual ir::BasicBlock execute()
    {
        auto seed  = generic::getDataAsInteger(getOperand(0));
        auto state = ir::data_cast<generic::RandomStateData>(getOperandData(1));

        matrix::swapDefaultRandomState(state.getRandomState());

        matrix::srand(seed);

        matrix::swapDefaultRandomState(state.getRandomState());

        return getParent();
    }

public:
    virtual std::shared_ptr<ValueImplementation> clone() const
    {
        return std::make_shared<SRandOperationImplementation>(*this);
    }

public:
    virtual std::string name() const
    {
        return "cpu-srand";
    }

public:
    virtual ir::Type getType() const
    {
        return ir::RandomStateType();
    }

};

SRandOperation::SRandOperation()
: TargetOperation(std::make_shared<SRandOperationImplementation>())
{

}

SRandOperation::~SRandOperation()
{

}

} // namespace cpu
} // namespace machine
} // namespace lucius








