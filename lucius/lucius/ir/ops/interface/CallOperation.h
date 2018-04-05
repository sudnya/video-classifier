/*  \file   CallOperation.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the CallOperation class.
*/

#pragma once

// Lucius Includes
#include <lucius/ir/ops/interface/ControlOperation.h>

// Forward Declarations
namespace lucius { namespace ir { class Function;         } }
namespace lucius { namespace ir { class ExternalFunction; } }

namespace lucius
{

namespace ir
{

/*! \brief A class for representing a function call. */
class CallOperation : public ControlOperation
{
public:
    explicit CallOperation(Function target);
    explicit CallOperation(ExternalFunction target);
    ~CallOperation();

};

} // namespace ir
} // namespace lucius



