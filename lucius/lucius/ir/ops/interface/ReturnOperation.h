/*  \file   ReturnOperation.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the ReturnOperation class.
*/

#pragma once

// Lucius Includes
#include <lucius/ir/ops/interface/ControlOperation.h>

namespace lucius
{

namespace ir
{

/*! \brief A class for representing a return from a function call. */
class ReturnOperation : public ControlOperation
{
public:
    ReturnOperation(Value returnedValue);
    ~ReturnOperation();

};

} // namespace ir
} // namespace lucius




