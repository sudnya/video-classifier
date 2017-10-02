/*  \file   TargetOperation.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the TargetOperation class.
*/

#pragma once

// Standard Library Includes
#include <list>

// Forward Declarations
namespace lucius { namespace ir { class ValueImplementation; } }
namespace lucius { namespace ir { class Operation;           } }
namespace lucius { namespace ir { class TargetValue;         } }
namespace lucius { namespace ir { class ShapeList;           } }
namespace lucius { namespace ir { class Type;                } }
namespace lucius { namespace ir { class PerformanceMetrics;  } }

namespace lucius
{

namespace ir
{

/*! \brief A class for representing an operation. */
class TargetOperation
{
public:
    TargetOperation();
    explicit TargetOperation(std::shared_ptr<ValueImplementation> value);
    ~TargetOperation();

public:
          Type& getOutputType();
    const Type& getOutputType() const;

public:
    PerformanceMetrics getPerformanceMetrics() const;

public:
    void execute();

public:
    void setOutputOperand(const TargetValue& v);

public:
    using TargetValueList = std::list<TargetValue>;

public:
          TargetValue& getOutputOperand();
    const TargetValue& getOutputOperand() const;

public:
          TargetValueList& getAllOperands();
    const TargetValueList& getAllOperands() const;

public:
    void setOperand(const TargetValue& v, size_t index);
    void appendOperand(const TargetValue& v);

public:
    std::shared_ptr<ValueImplementation> getValueImplementation() const;

};

} // namespace ir
} // namespace lucius

