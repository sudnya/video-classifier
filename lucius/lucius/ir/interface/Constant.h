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
namespace lucius { namespace ir { class ValueImplementation;    } }

namespace lucius
{

namespace ir
{

/*! \brief A class for representing a constant value a program. */
class Constant
{
public:
    Constant(std::shared_ptr<ValueImplementation>);

public:
    std::shared_ptr<ValueImplementation> getValueImplementation() const;
};

} // namespace ir
} // namespace lucius





