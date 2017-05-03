/*  \file   Context.h
    \author Gregory Diamos
    \date   April 22, 2017
    \brief  The header file for the Context class.
*/

#pragma once

namespace lucius
{

namespace lazy
{

/*! \brief Represents a lazily computed program. */
class Context
{
public:
    ir::Context& getContext();

public:
    void clear();

public:
    LazyValue getConstant(const Matrix& value);
    LazyValue getConstant(int64_t integer);

public:
    BasicBlock* newBasicBlock();
    void setBasicBlock(BasicBlock*);

public:
    IRBuilder& getBuilder();

private:
    std::unique_ptr<ContextImplementation> _implementation;
};

}

}



