/*  \file   TensorType.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the TensorType class.
*/

#pragma once

// Lucius Includes
#include <lucius/ir/interface/Type.h>

namespace lucius
{

namespace ir
{

/*! \brief A class for representing a type. */
class TensorType : public Type
{
public:
    TensorType(const Dimension& d, const Precision& p);
    ~TensorType();

};

} // namespace ir
} // namespace lucius



