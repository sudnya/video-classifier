/*  \file   TargetOperationImplementation.h
    \author Gregory Diamos
    \date   October 4, 2017
    \brief  The header file for the TargetOperationImplementation class.
*/

#pragma once

// Lucius Includes
#include <lucius/ir/implementation/ValueImplementation.h>

// Standard Library Includes
#include <list>

// Forward Declarations
namespace lucius { namespace ir { class Use;                } }
namespace lucius { namespace ir { class TargetValue;        } }
namespace lucius { namespace ir { class PerformanceMetrics; } }
namespace lucius { namespace ir { class BasicBlock;         } }

namespace lucius
{

namespace ir
{

/*! \brief The implementation of a class that represents an operation in the program. */
class TargetOperationImplementation : public ValueImplementation
{
public:
    const Use& getOperand(size_t index) const;
          Use& getOperand(size_t index);

public:
          Use& getOutputOperand();
    const Use& getOutputOperand() const;

public:
    using UseList = std::list<Use>;

public:
          UseList& getAllOperands();
    const UseList& getAllOperands() const;

public:
    void setOutputOperand(const TargetValue& v);

public:
    void setOperand(const TargetValue& v, size_t index);
    void appendOperand(const TargetValue& v);

public:
    /*! \brief Get the performance metrics for this operations. */
    virtual PerformanceMetrics getPerformanceMetrics() const = 0;

public:
    /*! \brief Execute the operation. */
    virtual BasicBlock execute() = 0;

};

} // namespace ir
} // namespace lucius






