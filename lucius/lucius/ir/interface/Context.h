/*  \file   Context.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the Context class.
*/

#pragma once

// Standard Library Includes
#include <memory>

// Forward Declarations
namespace lucius { namespace ir { class ContextImplementation; } }

namespace lucius
{

namespace ir
{

/*! \brief Holds 'global' state for the IR. */
class Context
{
private:
    std::unique_ptr<ContextImplementation> _implementation;

};

} // namespace ir
} // namespace lucius




