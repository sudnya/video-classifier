/*  \file   TensorType.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the TensorType class.
*/

#pragma once

// Lucius Includes
#include <lucius/ir/interface/Type.h>

// Forward Declarations
namespace lucius { namespace matrix { class Dimension; } }
namespace lucius { namespace matrix { class Precision; } }

namespace lucius { namespace ir { class Shape;              } }
namespace lucius { namespace ir { class TypeImplementation; } }

namespace lucius
{

namespace ir
{

/*! \brief A class for representing a type. */
class TensorType : public Type
{
public:
    using Precision = matrix::Precision;

public:
    TensorType(const Shape& d, const Precision& p);
    explicit TensorType(std::shared_ptr<TypeImplementation> );
    ~TensorType();

public:
    const Shape& getShape() const;
    const Precision& getPrecision() const;

};

} // namespace ir
} // namespace lucius



