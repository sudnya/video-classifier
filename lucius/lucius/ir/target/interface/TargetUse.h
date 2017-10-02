/*  \file   TargetUse.h
    \author Gregory Diamos
    \date   October 1, 2017
    \brief  The header file for the TargetUse class.
*/

#pragma once

// Standard Library Includes
#include <memory>

// Forward Declarations
namespace lucius { namespace ir { class TargetValue;     } }
namespace lucius { namespace ir { class BasicBlock;      } }
namespace lucius { namespace ir { class TargetOperation; } }

namespace lucius { namespace ir { class TargetUseImplementation; } }

namespace lucius
{

namespace ir
{

/*! \brief A class for representing and tracking the use of a value. */
class TargetUse
{
public:
          BasicBlock& getParent();
    const BasicBlock& getParent() const;

public:
          TargetOperation& getOperation();
    const TargetOperation& getOperation() const;

public:
          TargetValue& getValue();
    const TargetValue& getValue() const;

private:
    std::shared_ptr<TargetUseImplementation> _implementation;
};

} // namespace ir
} // namespace lucius






