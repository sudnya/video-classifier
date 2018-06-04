/*  \file   AddressType.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the AddressType class.
*/

#pragma once

// Lucius Includes
#include <lucius/ir/interface/Type.h>

namespace lucius
{

namespace ir
{

/*! \brief A class for representing a type. */
class AddressType : public Type
{
public:
    AddressType();
    ~AddressType();

};

} // namespace ir
} // namespace lucius





