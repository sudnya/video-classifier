/*  \file   AddressType.cpp
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The source file for the AddressType class.
*/

// Lucius Includes
#include <lucius/ir/types/interface/AddressType.h>

namespace lucius
{

namespace ir
{

AddressType::AddressType()
: Type(Type::PointerId)
{

}

AddressType::~AddressType()
{
    // intentionally blank
}

} // namespace ir
} // namespace lucius







