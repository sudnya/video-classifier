/*  \file   OperationImplementation.h
    \author Gregory Diamos
    \date   October 4, 2017
    \brief  The header file for the OperationImplementation class.
*/

#pragma once

// Lucius Includes
#include <lucius/ir/implementation/UserImplementation.h>

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

public:
    const Use& getOperand(size_t index) const;
          Use& getOperand(size_t index);

    const Shape& getOperandShape(size_t index) const;
          Shape& getOperandShape(size_t index);

public:
    const UseList& getOperands() const;
          UseList& getOperands();

public:
    void setOperands(const UseList& uses);

public:
    BasicBlock getParent() const;

public:
    using OperationList = std::list<Operation>;

    using operation_iterator = OperationList::iterator;
    using const_operation_iterator = OperationList::const_iterator;

public:
          operation_iterator getIterator();
    const_operation_iterator getIterator() const;

public:
    virtual std::string name() const = 0;

private:
    std::weak_ptr<BasicBlockImplementation> _parent;

private:
    operation_iterator _iterator;

};

} // namespace ir
} // namespace lucius





