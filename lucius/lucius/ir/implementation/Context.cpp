/*  \file   Context.cpp
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The source file for the Context class.
*/

// Lucius Includes
#include <lucius/ir/interface/Context.h>

#include <lucius/ir/interface/Type.h>
#include <lucius/ir/interface/Constant.h>

// Standard Library Includes
#include <list>

namespace lucius
{

namespace ir
{

class IdAllocator
{
public:
    IdAllocator()
    : _next(0)
    {

    }

public:
    size_t allocate()
    {
        return _next++;
    }

private:
    size_t _next;
};

class ContextImplementation
{

public:
    Constant addConstant(Constant constant)
    {
        _constants.push_back(constant);

        return _constants.back();
    }

    Type addType(Type type)
    {
        _types.push_back(type);

        return _types.back();
    }

    size_t allocateId()
    {
        return _idAllocator.allocate();
    }

private:
    using ConstantList = std::list<Constant>;
    using TypeList     = std::list<Type>;

private:
    ConstantList _constants;
    TypeList     _types;

private:
    IdAllocator _idAllocator;
};

Context::Context()
: _implementation(std::make_unique<ContextImplementation>())
{

}

Context::~Context()
{

}

Constant Context::addConstant(Constant constant)
{
    return _implementation->addConstant(constant);
}

Type Context::addType(Type type)
{
    return _implementation->addType(type);
}

size_t Context::allocateId()
{
    return _implementation->allocateId();
}

std::unique_ptr<Context> defaultContext;

Context& Context::getDefaultContext()
{
    if(!defaultContext)
    {
        defaultContext = std::make_unique<Context>();
    }

    return *defaultContext;
}

} // namespace ir
} // namespace lucius





