/*  \file   Gradient.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the Gradient class.
*/

#pragma once

// Lucius Includes
#include <lucius/ir/interface/Value.h>

namespace lucius
{

namespace ir
{

/*! \brief A class for a gradient value. */
class Gradient : public Value
{
public:
    Gradient();
    explicit Gradient(Value );
    explicit Gradient(Operation );
    ~Gradient();

};

} // namespace ir
} // namespace lucius





