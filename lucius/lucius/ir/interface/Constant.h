/*  \file   Constant.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the Constant class.
*/

#pragma once

// Standard Library Includes
#include <memory>

// Forward Declarations
namespace lucius { namespace ir { class ConstantImplementation; } }

namespace lucius
{

namespace ir
{

/*! \brief A class for representing a constant value a program. */
class Constant
{
public:
    std::shared_ptr<ConstantImplementation> _implementation;
};

} // namespace ir
} // namespace lucius





