/*  \file   ConstantImplementation.cpp
    \author Gregory Diamos
    \date   October 4, 2017
    \brief  The source file for the ConstantImplementation class.
*/

// Lucius Includes
#include <lucius/ir/implementation/ConstantImplementation.h>

// Standard Library Includes
#include <string>

namespace lucius
{

namespace ir
{

std::string ConstantImplementation::toSummaryString() const
{
    return toString();
}


} // namespace ir
} // namespace lucius







