/*  \file   Context.h
    \author Gregory Diamos
    \date   April 22, 2017
    \brief  The header file for the Context class.
*/

#pragma once

// Standard Library Includes
#include <cstdint>
#include <memory>

// Forward Declarations
namespace lucius { namespace ir { class Context;    } }
namespace lucius { namespace ir { class BasicBlock; } }
namespace lucius { namespace ir { class IRBuilder;  } }

namespace lucius { namespace lazy { class LazyValue;             } }
namespace lucius { namespace lazy { class ContextImplementation; } }

namespace lucius { namespace matrix { class Matrix; } }

namespace lucius
{

namespace lazy
{

/*! \brief Represents a lazily computed program. */
class Context
{
public:
    Context();
    ~Context();

public:
    ir::Context& getContext();

public:
    void clear();

public:
    using Matrix = matrix::Matrix;
    using BasicBlock = ir::BasicBlock;
    using IRBuilder = ir::IRBuilder;

public:
    LazyValue getConstant(const Matrix& value);
    LazyValue getConstant(int64_t integer);

public:
    BasicBlock newBasicBlock();
    void setBasicBlock(const BasicBlock& block);

public:
    IRBuilder& getBuilder();

private:
    std::unique_ptr<ContextImplementation> _implementation;
};

}

}



