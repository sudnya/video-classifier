/*  \file   ConstantAddress.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the ConstantAddress class.
*/

#pragma once

// Lucius Includes
#include <lucius/ir/interface/Constant.h>

namespace lucius
{

namespace ir
{

/*! \brief A class for representing a constant address value in a program. */
class ConstantAddress : public Constant
{
public:
    explicit ConstantAddress(void* value);
    explicit ConstantAddress(std::shared_ptr<ValueImplementation>);

public:
    void* getAddress() const;
};

} // namespace ir
} // namespace lucius








