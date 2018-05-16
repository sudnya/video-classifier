/*  \file   LazyIr.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the LazyIr interface functions.
*/

#pragma once

// Standard Library Includes
#include <cstdint>
#include <istream>
#include <vector>
#include <map>

// Forward Declarations
namespace lucius { namespace lazy { class Context;   } }
namespace lucius { namespace lazy { class LazyValue; } }

namespace lucius { namespace ir { class IRBuilder;  } }
namespace lucius { namespace ir { class BasicBlock; } }
namespace lucius { namespace ir { class Value;      } }

namespace lucius { namespace analysis { class Analysis; } }

namespace lucius { namespace matrix { class Matrix; } }

namespace lucius
{

namespace lazy
{

/*! \brief Creates a new thread local context used for lazy operations. */
void newThreadLocalContext();

/*! \brief Get the current thread local context. */
Context& getThreadLocalContext();

/*! \brief Save the context to a stream. */
void saveThreadLocalContext(std::ostream& stream);

/*! \brief Load the context from a stream. */
void loadThreadLocalContext(std::istream& stream);

/*! \brief Get the IR Builder. */
ir::IRBuilder& getBuilder();

/*! \brief Register a lazy value with the current context. */
void registerLazyValue(const LazyValue& value);

/*! \brief Get a lazy value constant representation of a Matrix. */
LazyValue getConstant(const matrix::Matrix& );

/*! \brief Get a lazy value constant representation of an integer. */
LazyValue getConstant(int64_t integer);

/*! \brief Create a new basic block. */
ir::BasicBlock newBasicBlock();

/*! \brief Set the current basic block that new operations will be added to. */
void setBasicBlock(const ir::BasicBlock& );

/*! \brief Get the handle to a lazy value */
size_t getHandle(LazyValue value);

/*! \brief Find the value with the specified handle. */
LazyValue lookupValueByHandle(size_t handle);

using MergedValueVector = std::vector<std::vector<ir::Value>>;
using ValueMap = std::map<ir::Value, ir::Value>;

/*! \brief Get the set of all lazy values. */
MergedValueVector getLazyValues(const ValueMap& mappedValues);

}

}

