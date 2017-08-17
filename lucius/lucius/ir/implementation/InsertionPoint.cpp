/*  \file   IRBuilder.cpp
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The source file for the InsertionPoint class.
*/

// Lucius Includes
#include <lucius/ir/interface/InsertionPoint.h>

#include <lucius/ir/interface/Operation.h>
#include <lucius/ir/interface/BasicBlock.h>

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

public:
    BasicBlock block;
    BasicBlock::iterator positionInBlock;
};

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


} // namespace ir
} // namespace lucius


