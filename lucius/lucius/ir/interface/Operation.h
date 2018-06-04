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
#include <list>

// Forward Declarations
namespace lucius { namespace ir { class OperationImplementation; } }
namespace lucius { namespace ir { class ValueImplementation;     } }
namespace lucius { namespace ir { class ShapeList;               } }
namespace lucius { namespace ir { class Use;                     } }
namespace lucius { namespace ir { class Type;                    } }
namespace lucius { namespace ir { class Context;                 } }
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
    ~Operation();

public:
    // forward shape operation
    ShapeList getOutputShapes(const ShapeList& inputShapes) const;

    // backward shape operation
    ShapeList getInputShapes(const ShapeList& outputShapes) const;

public:
    using UseList   = std::list<Use>;
    using ValueList = std::list<Value>;
    using OperationList = std::list<Operation>;

    using operation_iterator = OperationList::iterator;
    using const_operation_iterator = OperationList::const_iterator;

    using iterator = UseList::iterator;
    using const_iterator = UseList::const_iterator;

public:
          iterator begin();
    const_iterator begin() const;

          iterator end();
    const_iterator end() const;

public:
    const UseList& getOperands() const;
          UseList& getOperands();

public:
    size_t size() const;
    bool  empty() const;

public:
    const Use& getOperand(size_t index) const;
          Use& getOperand(size_t index);

public:
    void setOperands(const UseList& uses);
    void setOperands(const ValueList& values);

public:
    void appendOperand(const Use& use);
    void appendOperand(const Value& value);

public:
    void replaceOperand(const Use& original, const Use& newOperand);
    void insertOperand(iterator position, const Use& newOperand);


public:
    OperationList getPredecessors() const;
    OperationList getSuccessors() const;

public:
    ValueList getUsedValues() const;

public:
    /*! \brief Query whether or not the operation can change control flow. */
    bool isControlOperation() const;

    /*! \brief Query whether or not the operation is a function call. */
    bool isCall() const;

    /*! \brief Query whether or not the operation is a function return. */
    bool isReturn() const;

    /*! \brief Query whether or not the operation computes a gradient. */
    bool isGradientOperation() const;

    /*! \brief Query whether or not the operation is a PHI node. */
    bool isPHI() const;

    /*! \brief Query whether or not the operation is a target operation. */
    bool isTargetOperation() const;

    /*! \brief Query whether or not the operation returns a void type */
    bool isVoid() const;

    /*! \brief Query whether or not the operation is a load */
    bool isLoad() const;

public:
    Type getType() const;

public:
    BasicBlock getBasicBlock() const;
    BasicBlock getParent() const;
    void setParent(const BasicBlock& b);

public:
    /*! \brief Remove uses of operands. */
    void detach();
    /*! \brief Remove from parent basic block. */
    void detachFromBasicBlock();

public:
    void setIterator(operation_iterator it);

          operation_iterator getIterator();
    const_operation_iterator getIterator() const;

    size_t getIndexInBasicBlock() const;

public:
    Context* getContext();

public:
    Operation clone() const;

public:
    std::string name() const;

public:
    std::string toString() const;

public:
    std::shared_ptr<ValueImplementation> getValueImplementation() const;
    std::shared_ptr<OperationImplementation> getImplementation() const;

public:
    bool operator==(const Operation&) const;
    bool operator<(const Operation&) const;

private:
    std::shared_ptr<OperationImplementation> _implementation;

};

} // namespace ir
} // namespace lucius



