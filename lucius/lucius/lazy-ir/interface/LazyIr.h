/*  \file   LazyIr.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the LazyIr interface functions.
*/

#pragma once

// Standard Library Includes
#include <cstdint>

// Forward Declarations
namespace lucius { namespace lazy { class Context;   } }
namespace lucius { namespace lazy { class LazyValue; } }

namespace lucius { namespace ir { class IRBuilder;  } }
namespace lucius { namespace ir { class BasicBlock; } }

namespace lucius { namespace matrix { class Matrix; } }

namespace lucius
{

namespace lazy
{

/*! \brief Creates a new thread local context used for lazy operations. */
void newThreadLocalContext();

/*! \brief Get the current thread local context. */
Context& getThreadLocalContext();

/*! \brief Get the IR Builder. */
ir::IRBuilder& getBuilder();

/*! \brief Get a lazy value constant representation of a Matrix. */
LazyValue getConstant(const matrix::Matrix& );

/*! \brief Get a lazy value constant representation of an integer. */
LazyValue getConstant(int64_t integer);

/*! \brief Create a new basic block. */
ir::BasicBlock newBasicBlock();

/*! \brief Set the current basic block that new operations will be added to. */
void setBasicBlock(const ir::BasicBlock& );

}

}

