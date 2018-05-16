/*  \file   StructureType.cpp
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The source file for the StructureType class.
*/

// Lucius Includes
#include <lucius/ir/types/interface/StructureType.h>

#include <lucius/ir/implementation/TypeImplementation.h>

// Standard Library Includes
#include <string>
#include <sstream>

namespace lucius
{

namespace ir
{

class StructureTypeImplementation : public TypeImplementation
{
public:
    StructureTypeImplementation(std::initializer_list<Type> initializer)
    : TypeImplementation(Type::StructureId), _types(initializer)
    {

    }

public:
    StructureType::iterator begin()
    {
        return _types.begin();
    }

    StructureType::const_iterator begin() const
    {
        return _types.begin();
    }

    StructureType::iterator end()
    {
        return _types.end();
    }

    StructureType::const_iterator end() const
    {
        return _types.end();
    }

    const Type& getType(size_t position) const
    {
        return _types[position];
    }

    Type& getType(size_t position)
    {
        return _types[position];
    }

    size_t size() const
    {
        return _types.size();
    }

    bool empty() const
    {
        return _types.empty();
    }

public:
    virtual std::string toString() const
    {
        std::stringstream stream;

        stream << "structure {";

        auto type = begin();

        if(type != end())
        {
            stream << type->toString();

            ++type;
        }

        for(; type != end(); ++type)
        {
            stream << ", " << type->toString();
        }

        stream << "}";

        return stream.str();
    }

private:
    StructureType::TypeVector _types;
};

StructureType::StructureType()
: StructureType({})
{

}

StructureType::StructureType(std::initializer_list<Type> initializer)
: Type(std::make_shared<StructureTypeImplementation>(initializer))
{

}

StructureType::StructureType(std::shared_ptr<TypeImplementation> implementation)
: Type(implementation)
{

}

StructureType::~StructureType()
{
    // intentionally blank
}

StructureType::iterator StructureType::begin()
{
    return getImplementation()->begin();
}

StructureType::const_iterator StructureType::begin() const
{
    return getImplementation()->begin();
}

StructureType::iterator StructureType::end()
{
    return getImplementation()->end();
}

StructureType::const_iterator StructureType::end() const
{
    return getImplementation()->end();
}

const Type& StructureType::operator[](size_t position) const
{
    return getImplementation()->getType(position);
}

Type& StructureType::operator[](size_t position)
{
    return getImplementation()->getType(position);
}

size_t StructureType::size() const
{
    return getImplementation()->size();
}

bool StructureType::empty() const
{
    return getImplementation()->empty();
}

std::shared_ptr<StructureTypeImplementation> StructureType::getImplementation() const
{
    return std::static_pointer_cast<StructureTypeImplementation>(getTypeImplementation());
}


} // namespace ir
} // namespace lucius








