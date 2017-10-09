/*  \file   BinaryApplyOperation.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the BinaryApplyOperation class.
*/

#pragma once

// Lucius Includes
#include <lucius/ir/interface/Operation.h>

namespace lucius
{

namespace ir
{

/*! \brief A class for representing an operation. */
class BinaryApplyOperation : public Operation
{
public:
    BinaryApplyOperation(Value left, Value right, Value operation);
    ~BinaryApplyOperation();

};

} // namespace ir
} // namespace lucius





