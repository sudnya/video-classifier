/*  \file   Use.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the Use class.
*/

#pragma once

// Standard Library Includes
#include <memory>

// Forward Declarations
namespace lucius { namespace ir { class Value;      } }
namespace lucius { namespace ir { class BasicBlock; } }
namespace lucius { namespace ir { class Operation;  } }

namespace lucius { namespace ir { class UseImplementation; } }

namespace lucius
{

namespace ir
{

/*! \brief A class for representing and tracking the use of a value. */
class Use
{

public:
          BasicBlock& getParent();
    const BasicBlock& getParent() const;

public:
          Operation& getOperation();
    const Operation& getOperation() const;

public:
          Value& getValue();
    const Value& getValue() const;

private:
    std::shared_ptr<UseImplementation> _implementation;
};

} // namespace ir
} // namespace lucius






