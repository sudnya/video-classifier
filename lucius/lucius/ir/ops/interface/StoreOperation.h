/*  \file   StoreOperation.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the StoreOperation class.
*/

#pragma once

// Lucius Includes
#include <lucius/ir/interface/Operation.h>

// Forward Declaration
namespace lucius { namespace ir { class Variable; } }

namespace lucius
{
namespace ir
{

/*! \brief A class for representing a store operation. */
class StoreOperation : public Operation
{
public:
    StoreOperation(Variable variable, Value newValue);
    ~StoreOperation();

};

} // namespace ir
} // namespace lucius


