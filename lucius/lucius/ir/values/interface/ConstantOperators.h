/*  \file   ConstantOperators.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the ConstantOperators class.
*/

#pragma once

// Lucius Includes
#include <lucius/ir/interface/Constant.h>

namespace lucius
{

namespace ir
{

/*! \brief A class for representing a constant matrix value a program. */
class ConstantOperator : public Constant
{
public:
    explicit ConstantOperator(size_t operatorId);
};

class Add : public ConstantOperator
{
public:
    Add();

};

} // namespace ir
} // namespace lucius








