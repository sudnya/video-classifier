/*  \file   Context.h
    \author Gregory Diamos
    \date   April 22, 2017
    \brief  The header file for the Context class.
*/

#pragma once

// Standard Library Includes
#include <cstdint>
#include <memory>
#include <vector>
#include <map>

// Forward Declarations
namespace lucius { namespace ir { class Context;    } }
namespace lucius { namespace ir { class BasicBlock; } }
namespace lucius { namespace ir { class IRBuilder;  } }
namespace lucius { namespace ir { class Value;      } }

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
    using MergedValue = std::vector<ir::Value>;
    using MergedValueVector = std::vector<MergedValue>;
    using ValueMap = std::map<ir::Value, ir::Value>;

public:
    LazyValue getConstant(const Matrix& value);
    LazyValue getConstant(int64_t integer);

public:
    BasicBlock newBasicBlock();
    void setBasicBlock(const BasicBlock& block);

public:
    IRBuilder& getBuilder();

public:
    void registerLazyValue(const LazyValue& value);

public:
    MergedValueVector getLazyValues(const ValueMap& mappedValues);

private:
    std::unique_ptr<ContextImplementation> _implementation;
};

}

}



