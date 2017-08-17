/*  \file   RandomStateType.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the RandomStateType class.
*/

#pragma once

// Lucius Includes
#include <lucius/ir/interface/Type.h>

namespace lucius
{

namespace ir
{

/*! \brief A class for representing a type. */
class RandomStateType : public Type
{
public:
    RandomStateType();
    ~RandomStateType();

};

} // namespace ir
} // namespace lucius




