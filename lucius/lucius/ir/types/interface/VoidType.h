/*  \file   VoidType.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the VoidType class.
*/

#pragma once

// Lucius Includes
#include <lucius/ir/interface/Type.h>

namespace lucius
{

namespace ir
{

/*! \brief A class for representing a type. */
class VoidType : public Type
{
public:
    VoidType();
    ~VoidType();

};

} // namespace ir
} // namespace lucius





