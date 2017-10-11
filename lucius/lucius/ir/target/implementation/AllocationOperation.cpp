/*  \file   AllocationOperation.cpp
    \author Gregory Diamos
    \date   October 9, 2017
    \brief  The source file for the AllocationOperation class.
*/

// Lucius Includes
#include <lucius/ir/target/interface/AllocationOperation.h>

#include <lucius/ir/target/interface/TargetValue.h>

namespace lucius
{

namespace ir
{

AllocationOperation::AllocationOperation(TargetValue allocatedValue)
{
    setOutputOperand(allocatedValue);
}

AllocationOperation::~AllocationOperation()
{
    // intentionally blank
}

} // namespace ir
} // namespace lucius



