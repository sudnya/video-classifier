/*  \file   TargetOperation.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the TargetOperation class.
*/

#pragma once

// Lucius Includes
#include <lucius/util/interface/IteratorRange.h>

// Standard Library Includes
#include <list>

// Forward Declarations
namespace lucius { namespace ir { class TargetOperationImplementation; } }
namespace lucius { namespace ir { class ValueImplementation;           } }
namespace lucius { namespace ir { class Operation;                     } }
namespace lucius { namespace ir { class BasicBlock;                    } }
namespace lucius { namespace ir { class TargetValue;                   } }
namespace lucius { namespace ir { class Use;                           } }
namespace lucius { namespace ir { class ShapeList;                     } }
namespace lucius { namespace ir { class Type;                          } }
namespace lucius { namespace ir { class PerformanceMetrics;            } }

namespace lucius
{

namespace ir
{

/*! \brief A class for representing an operation. */
class TargetOperation
{
public:
    TargetOperation();
    explicit TargetOperation(std::shared_ptr<ValueImplementation> op);
    ~TargetOperation();

public:
    Type getOutputType() const;

public:
    PerformanceMetrics getPerformanceMetrics() const;

public:
    void execute();

public:
    void setOutputOperand(const TargetValue& v);

public:
    using UseList = std::list<Use>;

    using iterator = UseList::iterator;
    using const_iterator = UseList::const_iterator;

    using IteratorRange = util::IteratorRange<iterator>;
    using ConstIteratorRange = util::IteratorRange<const_iterator>;

public:
          Use& getOutputOperand();
    const Use& getOutputOperand() const;

          iterator getOutputOperandPosition();
    const_iterator getOutputOperandPosition() const;

    bool hasOutputOperand() const;

public:
          Use& getOperand(size_t index);
    const Use& getOperand(size_t index) const;

          iterator getOperandPosition(size_t index);
    const_iterator getOperandPosition(size_t index) const;

public:
          iterator begin();
    const_iterator begin() const;

          iterator end();
    const_iterator end() const;

public:
         IteratorRange getInputOperandRange();
    ConstIteratorRange getInputOperandRange() const;

public:
          UseList& getOperands();
    const UseList& getOperands() const;

    bool hasInputOperands() const;

public:
    void setOperand(const TargetValue& v, size_t index);
    void appendOperand(const TargetValue& v);
    void replaceOperand(const Use& original, const Use& newOperand);

public:
    size_t getInputOperandCount() const;

public:
    bool isValid() const;
    bool isCall() const;
    bool isReturn() const;
    bool isPHI() const;

public:
    BasicBlock getBasicBlock() const;

public:
    std::string toString() const;

public:
    std::shared_ptr<TargetOperationImplementation> getTargetOperationImplementation() const;

public:
    std::shared_ptr<ValueImplementation> getValueImplementation() const;

private:
    std::shared_ptr<TargetOperationImplementation> _implementation;

};

} // namespace ir
} // namespace lucius

