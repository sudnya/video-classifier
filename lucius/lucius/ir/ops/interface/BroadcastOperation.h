/*  \file   BroadcastOperation.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the BroadcastOperation class.
*/

#pragma once

// Lucius Includes
#include <lucius/ir/interface/Operation.h>

namespace lucius
{

namespace ir
{

/*! \brief A class for representing a broadcast operation. */
class BroadcastOperation : public Operation
{
public:
    BroadcastOperation(Value left, Value right, Value dimensions, Value operation);
    ~BroadcastOperation();

};

} // namespace ir
} // namespace lucius







