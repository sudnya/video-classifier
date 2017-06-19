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
        FunctionTypeId, // Functions
        StructureId,    // Structures
        ArrayId,        // Arrays
        PointerId,      // Pointers
        TensorId        // Tensors
    };

private:
    std::shared_ptr<TypeImplementation> _implementation;

};

} // namespace ir
} // namespace lucius


