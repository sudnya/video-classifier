/*  \file   FreeOperation.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the FreeOperation class.
*/

// Lucius Includes
#include <lucius/ir/target/interface/FreeOperation.h>

#include <lucius/ir/target/interface/TargetValue.h>

namespace lucius
{

namespace ir
{

FreeOperation::FreeOperation(TargetValue allocatedValue)
{
    appendOperand(allocatedValue);
}

FreeOperation::~FreeOperation()
{

}

} // namespace ir
} // namespace lucius




