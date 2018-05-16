/*  \file   StructureType.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the StructureType class.
*/

#pragma once

// Lucius Includes
#include <lucius/ir/interface/Type.h>

// Standard Library Includes
#include <vector>

// Forward Declarations
namespace lucius { namespace ir { class          TypeImplementation; } }
namespace lucius { namespace ir { class StructureTypeImplementation; } }

namespace lucius
{

namespace ir
{

/*! \brief A class for representing a structure type. */
class StructureType : public Type
{
public:
    StructureType();
    explicit StructureType(std::initializer_list<Type>);
    explicit StructureType(std::shared_ptr<TypeImplementation> );
    ~StructureType();

public:
    using TypeVector = std::vector<Type>;

    using iterator = TypeVector::iterator;
    using const_iterator = TypeVector::const_iterator;

public:
          iterator begin();
    const_iterator begin() const;

          iterator end();
    const_iterator end() const;

public:
    const Type& operator[](size_t position) const;
          Type& operator[](size_t position);

public:
    size_t size() const;
    bool  empty() const;

public:
    std::shared_ptr<StructureTypeImplementation> getImplementation() const;

};

} // namespace ir
} // namespace lucius







