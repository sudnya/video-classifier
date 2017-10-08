/*  \file   ConstantInteger.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the ConstantInteger class.
*/

#pragma once

// Lucius Includes
#include <lucius/ir/interface/Constant.h>

namespace lucius
{

namespace ir
{

/*! \brief A class for representing a constant scalar integer value a program. */
class ConstantInteger : public Constant
{
public:
    explicit ConstantInteger(size_t value);

public:
    size_t getValue() const;
};

} // namespace ir
} // namespace lucius







