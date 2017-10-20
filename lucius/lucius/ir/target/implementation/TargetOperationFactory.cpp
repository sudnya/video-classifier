/*  \file   TargetOperationFactory.cpp
    \author Gregory Diamos
    \date   August 16, 2017
    \brief  The source file for the TargetOperationFactory class.
*/

// Lucius Includes
#include <lucius/ir/target/interface/TargetOperationFactory.h>

#include <lucius/ir/target/interface/TargetOperation.h>

namespace lucius
{

namespace ir
{

TargetOperation TargetOperationFactory::create(const std::string& name)
{
    throw std::runtime_error("There is no registered target operation with name '" + name + "'.");
}

} // namespace ir
} // namespace lucius



