/*  \file   Operation.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the Operation class.
*/

#pragma once

// Lucius Includes
#include <lucius/ir/interface/User.h>

// Standard Library Includes
#include <string>
#include <vector>

// Forward Declarations
namespace lucius { namespace ir { class OperationImplementation; } }
namespace lucius { namespace ir { class ValueImplementation;     } }
namespace lucius { namespace ir { class ShapeList;               } }
namespace lucius { namespace ir { class Use;                     } }
namespace lucius { namespace ir { class Value;                   } }
namespace lucius { namespace ir { class BasicBlock;              } }
namespace lucius { namespace ir { class TargetOperation;         } }

namespace lucius
{

namespace ir
{

/*! \brief A class for representing an operation. */
class Operation
{
public:
    Operation();
    explicit Operation(std::shared_ptr<ValueImplementation> );
    Operation(const TargetOperation& op );
    virtual ~Operation();

public:
    // forward shape operation
    virtual ShapeList getOutputShapes(const ShapeList& inputShapes) const;

    // backward shape operation
    virtual ShapeList getInputShapes(const ShapeList& outputShapes) const;

public:
    using UseList   = std::list<Use>;
    using ValueList = std::list<Value>;
    using OperationList = std::list<Operation>;

    using operation_iterator = OperationList::iterator;
    using const_operation_iterator = OperationList::const_iterator;

public:
    const UseList& getOperands() const;
          UseList& getOperands();

public:
    const Use& getOperand(size_t index) const;
          Use& getOperand(size_t index);

public:
    void setOperands(const UseList& uses);
    void setOperands(const ValueList& values);

public:
    OperationList getPredecessors() const;
    OperationList getSuccessors() const;

public:
    ValueList getValues() const;

public:
    /*! \brief Query whether or not the operation can change control flow. */
    bool isControlOperation() const;

    /*! \brief Query whether or not the operation computes a gradient. */
    bool isGradientOperation() const;

public:
          Type& getType();
    const Type& getType() const;

public:
          BasicBlock& getParent();
    const BasicBlock& getParent() const;

public:
          operation_iterator getIterator();
    const_operation_iterator getIterator() const;

public:
    Operation clone() const;

public:
    std::shared_ptr<ValueImplementation> getValueImplementation() const;

public:
    bool operator==(const Operation&) const;
    bool operator<(const Operation&) const;

private:
    std::shared_ptr<OperationImplementation> _implementation;

};

} // namespace ir
} // namespace lucius



