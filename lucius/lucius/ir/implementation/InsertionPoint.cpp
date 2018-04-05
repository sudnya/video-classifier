/*  \file   IRBuilder.cpp
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The source file for the InsertionPoint class.
*/

// Lucius Includes
#include <lucius/ir/interface/InsertionPoint.h>

#include <lucius/ir/interface/Operation.h>
#include <lucius/ir/interface/BasicBlock.h>
#include <lucius/ir/interface/Function.h>

namespace lucius
{
namespace ir
{

class InsertionPointImplementation
{
public:
    InsertionPointImplementation()
    {

    }

    InsertionPointImplementation(Operation operation)
    : block(operation.getParent()), positionInBlock(operation.getIterator())
    {

    }

    InsertionPointImplementation(BasicBlock block)
    : block(block), positionInBlock(block.begin())
    {

    }

    InsertionPointImplementation(BasicBlock block, BasicBlock::iterator position)
    : block(block), positionInBlock(position)
    {

    }

public:
    BasicBlock block;
    BasicBlock::iterator positionInBlock;
};

InsertionPoint::InsertionPoint(const BasicBlock& block, BasicBlock::iterator position)
: _implementation(std::make_unique<InsertionPointImplementation>(block, position))
{

}

InsertionPoint::InsertionPoint(const BasicBlock& block)
: _implementation(std::make_unique<InsertionPointImplementation>(block))
{

}

InsertionPoint::InsertionPoint(const Operation& operation)
: _implementation(std::make_unique<InsertionPointImplementation>(operation))
{

}

InsertionPoint::InsertionPoint()
: _implementation(std::make_unique<InsertionPointImplementation>())
{

}

InsertionPoint::InsertionPoint(const InsertionPoint& p)
: _implementation(std::make_unique<InsertionPointImplementation>(*p._implementation))
{

}

InsertionPoint::~InsertionPoint()
{
    // intentionally blank
}

InsertionPoint& InsertionPoint::operator=(const InsertionPoint& p)
{
    *_implementation = *p._implementation;

    return *this;
}

BasicBlock& InsertionPoint::getBasicBlock()
{
    return _implementation->block;
}

const BasicBlock& InsertionPoint::getBasicBlock() const
{
    return _implementation->block;
}

InsertionPoint::OperationList::iterator& InsertionPoint::getIterator()
{
    return _implementation->positionInBlock;
}

const InsertionPoint::OperationList::iterator& InsertionPoint::getIterator() const
{
    return _implementation->positionInBlock;
}

Function InsertionPoint::getFunction() const
{
    return getBasicBlock().getFunction();
}

} // namespace ir
} // namespace lucius


