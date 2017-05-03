/*  \file   Context.cpp
    \author Gregory Diamos
    \date   April 22, 2017
    \brief  The source file for the Context class.
*/

namespace lucius
{

namespace lazy
{

class ContextImplementation
{
public:
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
    BasicBlock* newBasicBlock()
    {
        return _builder.newBasicBlock();
    }

    void setBasicBlock(BasicBlock* block)
    {
        _builder.setInsertionPoint(block);
    }

public:
    LazyValue getContstant(const Matrix& value)
    {
        return _builder.getConstant(value);
    }

    LazyValue getConstant(int64_t integer)
    {
        return _builder.getConstant(integer);
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

BasicBlock* Context::newBasicBlock()
{
    return _implementation->newBasicBlock();
}

void Context::setBasicBlock(BasicBlock* block)
{
    _implementation->setBasicBlock(block);
}

IRBuilder& Context::getBuilder()
{
    return _implementation->getBuilder();
}

}

}




