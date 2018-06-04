/*  \file   SetOperation.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the SetOperation class.
*/

#pragma once

// Lucius Includes
#include <lucius/ir/interface/Operation.h>

namespace lucius
{

namespace ir
{

/*! \brief A class for representing a Set operation. */
class SetOperation : public Operation
{
public:
    SetOperation(Value value, Value shape);
    SetOperation();
    ~SetOperation();

};

} // namespace ir
} // namespace lucius






