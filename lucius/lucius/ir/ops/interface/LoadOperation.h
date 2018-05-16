/*  \file   LoadOperation.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the LoadOperation class.
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

/*! \brief A class for representing a load operation. */
class LoadOperation : public Operation
{
public:
    LoadOperation(Variable newValue);
    ~LoadOperation();

};

} // namespace ir
} // namespace lucius



