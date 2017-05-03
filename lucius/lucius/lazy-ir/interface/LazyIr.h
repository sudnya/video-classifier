/*  \file   LazyIr.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the LazyIr interface functions.
*/

#pragma once

// Forward Declarations
namespace lucius { namespace lazy { class Context; } }

namespace lucius
{

namespace lazy
{

/*! \brief Creates a new thread local context used for lazy operations. */
void newThreadLocalContext();

/*! \brief Get the current thread local context. */
Context& getThreadLocalContext();

/*! \brief Get the IR Builder. */
IRBuilder& getBuilder();

/*! \brief Get a lazy value constant representation of a Matrix. */
LazyValue getConstant(const Matrix& );

/*! \brief Get a lazy value constant representation of an integer. */
LazyValue getConstant(int64_t integer);

/*! \brief Create a new basic block. */
BasicBlock* newBasicBlock();

/*! \brief Set the current basic block that new operations will be added to. */
void setBasicBlock(BasicBlock* );

}

}

