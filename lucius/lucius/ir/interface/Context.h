/*  \file   Context.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the Context class.
*/

#pragma once

// Forward Declarations
namespace lucius { namespace ir { class ContextImplementation; } }

namespace lucius
{

namespace ir
{

/*! \brief Holds 'global' state for the IR. */
class Context
{
public:

private:
    std::unique_ptr<ContextImplementation> _implementation;

};

} // namespace ir
} // namespace lucius




