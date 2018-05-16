/*  \file   LazyValue.h
    \author Gregory Diamos
    \date   April 22, 2017
    \brief  The header file for the LazyValue class.
*/

#pragma once

// Standard Library Includes
#include <memory>
#include <set>

// Forward Declarations
namespace lucius { namespace ir { class Value;    } }
namespace lucius { namespace ir { class Variable; } }

namespace lucius { namespace matrix { class Matrix; } }

namespace lucius { namespace lazy { class LazyValueImplementation; } }

namespace lucius
{

namespace lazy
{

/*! \brief Represents the result of a lazy computation. */
class LazyValue
{
public:
    LazyValue();
    explicit LazyValue(ir::Value );
    explicit LazyValue(ir::Variable );

public:
    template <typename T>
    T materialize();

    matrix::Matrix materialize();

public:
    ir::Value getValueForRead();
    ir::Value getValue() const;

public:
    bool isVariable() const;

public:
    using ValueSet = std::set<ir::Value>;

    const ValueSet& getDefinitions() const;

public:
    void addDefinition(ir::Value newDefinition);

public:
    bool operator<(const LazyValue& right) const;

private:
    void* _runProgram();
    void _clearState();

private:
    std::shared_ptr<LazyValueImplementation> _implementation;

};

}

}

#include <lucius/lazy-ir/implementation/LazyValue.inl>

