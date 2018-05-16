/*  \file   OperationImplementation.h
    \author Gregory Diamos
    \date   October 4, 2017
    \brief  The header file for the OperationImplementation class.
*/

#pragma once

// Lucius Includes
#include <lucius/ir/implementation/UserImplementation.h>
#include <lucius/ir/implementation/ValueImplementation.h>

// Forward Declarations
namespace lucius { namespace ir { class ShapeList;                } }
namespace lucius { namespace ir { class Shape;                    } }
namespace lucius { namespace ir { class Use;                      } }
namespace lucius { namespace ir { class BasicBlock;               } }
namespace lucius { namespace ir { class Operation;                } }
namespace lucius { namespace ir { class BasicBlockImplementation; } }

namespace lucius
{

namespace ir
{

/*! \brief The implementation of a class that represents an operation in the program. */
class OperationImplementation : public UserImplementation, public ValueImplementation
{
public:
    // forward shape operation
    virtual ShapeList getOutputShapes(const ShapeList& inputShapes) const;

    // backward shape operation
    virtual ShapeList getInputShapes(const ShapeList& outputShapes) const;

public:
    using UseList = std::list<Use>;
    using ValueList = std::list<Value>;

    using iterator = UseList::iterator;
    using const_iterator = UseList::const_iterator;

public:
    const Use& getOperand(size_t index) const;
          Use& getOperand(size_t index);

    const Shape& getOperandShape(size_t index) const;
          Shape& getOperandShape(size_t index);

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
    const UseList& getOperands() const;
          UseList& getOperands();

public:
          iterator begin();
    const_iterator begin() const;

          iterator end();
    const_iterator end() const;

public:
    size_t size() const;
    bool  empty() const;

public:
    BasicBlock getParent() const;
    BasicBlock getBasicBlock() const;
    void setParent(const BasicBlock&);

public:
    using OperationList = std::list<Operation>;

    using operation_iterator = OperationList::iterator;
    using const_operation_iterator = OperationList::const_iterator;

public:
    void setIterator(operation_iterator it);

          operation_iterator getIterator();
    const_operation_iterator getIterator() const;

public:
    void setImplementation(std::weak_ptr<OperationImplementation> implementation);
    std::shared_ptr<OperationImplementation> getImplementation() const;

public:
    virtual std::string name() const = 0;

public:
    virtual std::string toString() const;
    virtual std::string toSummaryString() const;
    std::string operandString() const;

private:
    std::weak_ptr<OperationImplementation> _this;

private:
    std::weak_ptr<BasicBlockImplementation> _parent;

private:
    operation_iterator _iterator;

};

} // namespace ir
} // namespace lucius





