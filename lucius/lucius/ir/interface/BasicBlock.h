/*  \file   BasicBlock.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the BasicBlock class.
*/

#pragma once

// Standard Library Includes
#include <memory>

// Forward Declarations
namespace lucius { namespace ir { class BasicBlockImplementation; } }

namespace lucius
{

namespace ir
{


/*! \brief A class for representing a basic block of operations in a program. */
class BasicBlock
{
private:
    std::shared_ptr<BasicBlockImplementation> _implementation;
};

} // namespace ir
} // namespace lucius




