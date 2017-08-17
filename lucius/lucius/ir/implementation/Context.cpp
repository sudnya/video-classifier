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

private:
    using ConstantList = std::list<Constant>;
    using TypeList     = std::list<Type>;

private:
    ConstantList _constants;
    TypeList     _types;
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

} // namespace ir
} // namespace lucius





