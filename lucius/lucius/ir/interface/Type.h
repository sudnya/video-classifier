/*  \file   Type.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the Type class.
*/

#pragma once

// Standard Library Includes
#include <memory>

// Forward Declarations
namespace lucius { namespace ir { class TypeImplementation; } }

namespace lucius
{

namespace ir
{

/*! \brief A class for representing a type. */
class Type
{
public:
    enum TypeId
    {
        VoidId,    // type with no size
        HalfId,    // 16-bit floating point
        FloatId,   // 32-bit floating point
        DoubleId,  // 64-bit floating point

        // Derived types
        IntegerId,      // Arbitrary precision integer
        FunctionId,     // Functions
        StructureId,    // Structures
        ArrayId,        // Arrays
        PointerId,      // Pointers
        TensorId,       // Tensors
        BasicBlockId    // Basic Block
    };

public:
    Type(TypeId);
    Type(std::shared_ptr<TypeImplementation>);
    Type();
    ~Type();

public:
    /*! \brief Is the type a scalar type. */
    bool isScalar() const;

    /*! \brief Is the type a tensor type. */
    bool isTensor() const;

    /*! \brief Is the type a tensor type. */
    bool isInteger() const;

    /*! \brief Is the type a float type. */
    bool isFloat() const;

    /*! \brief Is the type a pointer type. */
    bool isPointer() const;

    /*! \brief Is the type void */
    bool isVoid() const;

    /*! \brief Is the type a basic block. */
    bool isBasicBlock() const;

    /*! \brief Get the number of bytes needed to represent the type. */
    size_t getBytes() const;

public:
    std::string toString() const;

public:
    std::shared_ptr<TypeImplementation> getTypeImplementation() const;

private:
    std::shared_ptr<TypeImplementation> _implementation;

};

template <typename T, typename V>
T type_cast(const V& v)
{
    return T(v.getTypeImplementation());
}

} // namespace ir
} // namespace lucius


