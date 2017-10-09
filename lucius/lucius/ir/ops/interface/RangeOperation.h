/*  \file   RangeOperation.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the RangeOperation class.
*/

#pragma once

// Lucius Includes
#include <lucius/ir/interface/Operation.h>

namespace lucius
{

namespace ir
{

/*! \brief A class for representing a operation that creates an array of a range of values. */
class RangeOperation : public Operation
{
public:
    RangeOperation(Type tensorType);
    ~RangeOperation();

};

} // namespace ir
} // namespace lucius



