/*  \file   Argument.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the Argument class.
*/

#pragma once

// Standard Library Includes
#include <vector>

// Forward Declarations
namespace lucius { namespace ir { class Type; } }

namespace lucius
{

namespace ir
{

/*! \brief A class for representing a function argument. */
class Argument
{
public:
    const Type* getType() const;

private:
    const Type* _type;
};

typedef std::vector<Argument> ArgumentList;

} // namespace ir
} // namespace lucius

