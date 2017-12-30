/*  \file   ConstantImplementation.h
    \author Gregory Diamos
    \date   October 4, 2017
    \brief  The header file for the ConstantImplementation class.
*/

#pragma once

// Lucius Includes
#include <lucius/ir/implementation/ValueImplementation.h>

namespace lucius
{

namespace ir
{

/*! \brief The implementation of a class that represents a constant value in the program. */
class ConstantImplementation : public ValueImplementation
{
public:
    virtual std::string toSummaryString() const;

};

} // namespace ir
} // namespace lucius






