/*  \file   PHIOperation.cpp
    \author Gregory Diamos
    \date   October 11, 2017
    \brief  The source file for the PHIOperation class.
*/

// Lucius Includes
#include <lucius/ir/ops/interface/PHIOperation.h>

#include <lucius/ir/interface/BasicBlock.h>

#include <lucius/util/interface/debug.h>

namespace lucius
{

namespace ir
{

PHIOperation::PHIOperation(std::shared_ptr<ValueImplementation> implementation)
: Operation(implementation)
{

}

PHIOperation::~PHIOperation()
{

}

PHIOperation::BasicBlockVector PHIOperation::getPredecessorBasicBlocks() const
{
    assertM(false, "Not implemented.");

    return {};
}

} // namespace ir
} // namespace lucius






