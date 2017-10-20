/*  \file   ControlOperation.cpp
    \author Gregory Diamos
    \date   October 11, 2017
    \brief  The source file for the ControlOperation class.
*/

// Lucius Includes
#include <lucius/ir/ops/interface/ControlOperation.h>

namespace lucius
{

namespace ir
{

ControlOperation::ControlOperation(std::shared_ptr<ValueImplementation> implementation)
: Operation(implementation)
{

}

ControlOperation::~ControlOperation()
{

}

} // namespace ir
} // namespace lucius





