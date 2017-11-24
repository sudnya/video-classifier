/*  \file   Gradient.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the Gradient class.
*/

#pragma once

// Standard Library Includes
#include <memory>

// Forward Declarations
namespace lucius { namespace ir { class Value;                   } }
namespace lucius { namespace ir { class Operation;               } }
namespace lucius { namespace ir { class OperationImplementation; } }
namespace lucius { namespace ir { class ValueImplementation;     } }

namespace lucius
{

namespace ir
{

/*! \brief A class for a gradient value that should be computed. */
class Gradient
{
public:
    Gradient();
    explicit Gradient(Value );
    explicit Gradient(Operation );
    ~Gradient();

public:
    std::shared_ptr<ValueImplementation> getValueImplementation();

public:
    std::shared_ptr<OperationImplementation> _implementation;

};

} // namespace ir
} // namespace lucius





