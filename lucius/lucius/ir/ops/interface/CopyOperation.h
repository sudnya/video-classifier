/*  \file   CopyOperation.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the CopyOperation class.
*/

#pragma once

// Lucius Includes
#include <lucius/ir/interface/Operation.h>

namespace lucius
{

namespace ir
{

/*! \brief A class for representing a copy operation. */
class CopyOperation : public Operation
{
public:
    CopyOperation(Value input);
    ~CopyOperation();

};

} // namespace ir
} // namespace lucius

