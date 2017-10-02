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

namespace lucius
{

namespace ir
{

/*! \brief A class for representing a type. */
class TensorType : public Type
{
public:
    using Dimension = matrix::Dimension;
    using Precision = matrix::Precision;

public:
    TensorType(const Dimension& d, const Precision& p);
    ~TensorType();

};

} // namespace ir
} // namespace lucius



