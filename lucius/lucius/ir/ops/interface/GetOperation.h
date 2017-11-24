/*  \file   GetOperation.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the GetOperation class.
*/

#pragma once

// Lucius Includes
#include <lucius/ir/interface/Operation.h>

namespace lucius
{

namespace ir
{

/*! \brief A class for representing a Get operation. */
class GetOperation : public Operation
{
public:
    GetOperation(Value container, Value position);
    ~GetOperation();

};

} // namespace ir
} // namespace lucius





