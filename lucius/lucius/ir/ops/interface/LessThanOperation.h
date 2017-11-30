/*  \file   LessThanOperation.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the LessThanOperation class.
*/

#pragma once

// Lucius Includes
#include <lucius/ir/interface/Operation.h>

namespace lucius
{

namespace ir
{

/*! \brief A class for representing a LessThan operation. */
class LessThanOperation : public Operation
{
public:
    LessThanOperation(Value container, Value position);
    ~LessThanOperation();

};

} // namespace ir
} // namespace lucius






