/*  \file   ConstantFloat.h
    \author Gregory Diamos
    \date   April 4, 2018
    \brief  The header file for the ConstantFloat class.
*/

#pragma once

// Lucius Includes
#include <lucius/ir/interface/Constant.h>

namespace lucius
{

namespace ir
{

/*! \brief A class for representing a constant scalar integer value a program. */
class ConstantFloat : public Constant
{
public:
    explicit ConstantFloat(float value);
    explicit ConstantFloat(std::shared_ptr<ValueImplementation>);

public:
    float getValue() const;
};

} // namespace ir
} // namespace lucius








