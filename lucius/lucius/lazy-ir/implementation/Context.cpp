/*  \file   Context.cpp
    \author Gregory Diamos
    \date   April 22, 2017
    \brief  The source file for the Context class.
*/

// Lucius Includes
#include <lucius/lazy-ir/interface/Context.h>
#include <lucius/lazy-ir/interface/LazyValue.h>

#include <lucius/ir/interface/Context.h>
#include <lucius/ir/interface/IRBuilder.h>
#include <lucius/ir/interface/BasicBlock.h>
#include <lucius/ir/interface/Constant.h>

namespace lucius
{

namespace lazy
{

// Namespace Imports
using BasicBlock = ir::BasicBlock;
using IRBuilder = ir::IRBuilder;
using Matrix = matrix::Matrix;

class ContextImplementation
{
public:
    ContextImplementation()
    : _builder(getContext())
    {

    }

    ir::Context& getContext()
    {
        return _context;
    }

public:
    void clear()
    {
        _builder.clear();
    }

public:
    BasicBlock newBasicBlock()
    {
        return _builder.addBasicBlock();
    }

    void setBasicBlock(const BasicBlock& block)
    {
        _builder.setInsertionPoint(block);
    }

public:
    LazyValue getConstant(const Matrix& value)
    {
        return LazyValue(_builder.addConstant(value));
    }

    LazyValue getConstant(int64_t integer)
    {
        return LazyValue(_builder.addConstant(integer));
    }

public:
    IRBuilder& getBuilder()
    {
        return _builder;
    }

private:
    ir::Context _context;

private:
    ir::IRBuilder _builder;

};

Context::Context()
: _implementation(std::make_unique<ContextImplementation>())
{

}

Context::~Context()
{
    // intentionally blank
}

ir::Context& Context::getContext()
{
    return _implementation->getContext();
}

void Context::clear()
{
    _implementation->clear();
}

LazyValue Context::getConstant(const Matrix& value)
{
    return _implementation->getConstant(value);
}

LazyValue Context::getConstant(int64_t integer)
{
    return _implementation->getConstant(integer);
}

BasicBlock Context::newBasicBlock()
{
    return _implementation->newBasicBlock();
}

void Context::setBasicBlock(const BasicBlock& block)
{
    _implementation->setBasicBlock(block);
}

IRBuilder& Context::getBuilder()
{
    return _implementation->getBuilder();
}

}

}




