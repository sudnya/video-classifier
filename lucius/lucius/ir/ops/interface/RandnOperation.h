/*  \file   RandnOperation.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the RandOperation class.
*/

#pragma once

// Lucius Includes
#include <lucius/ir/interface/Operation.h>

namespace lucius
{

namespace ir
{

/*! \brief An operation that produces a random normal array of data and updates random state. */
class RandnOperation : public Operation
{
public:
    RandnOperation(Value state, Type tensorType);
    ~RandnOperation();

};

} // namespace ir
} // namespace lucius

