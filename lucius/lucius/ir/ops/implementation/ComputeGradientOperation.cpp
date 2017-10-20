/*  \file   ComputeGradientOperation.cpp
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The source file for the ComputeGradientOperation class.
*/

// Lucius Includes
#include <lucius/ir/ops/interface/ComputeGradientOperation.h>

#include <lucius/ir/interface/Use.h>
#include <lucius/ir/interface/Value.h>

#include <lucius/ir/ops/implementation/ComputeGradientOperationImplementation.h>

namespace lucius
{

namespace ir
{

ComputeGradientOperation::ComputeGradientOperation(Value value, Value cost)
: Operation(std::make_shared<ComputeGradientOperationImplementation>())
{
    setOperands({value, cost});
}

ComputeGradientOperation::ComputeGradientOperation(
    std::shared_ptr<ValueImplementation> implementation)
: Operation(implementation)
{

}

ComputeGradientOperation::~ComputeGradientOperation()
{

}

} // namespace ir
} // namespace lucius



