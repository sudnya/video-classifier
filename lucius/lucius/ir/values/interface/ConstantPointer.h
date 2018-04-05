/*  \file   ConstantPointer.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the ConstantPointer class.
*/

#pragma once

// Lucius Includes
#include <lucius/ir/interface/Constant.h>

namespace lucius
{

namespace ir
{

/*! \brief A class for representing a constant scalar integer value a program. */
class ConstantPointer : public Constant
{
public:
    explicit ConstantPointer(void* value);
    explicit ConstantPointer(std::shared_ptr<ValueImplementation>);

public:
    void* getValue() const;
};

} // namespace ir
} // namespace lucius








