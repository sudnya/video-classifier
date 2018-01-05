/*  \file   TensorValue.h
    \author Gregory Diamos
    \date   October 11, 2017
    \brief  The header file for the TensorValue class.
*/

#pragma once

// Lucius Includes
#include <lucius/ir/interface/Value.h>

// Forward Declarations
namespace lucius { namespace ir { class Shape; } }

namespace lucius { namespace matrix { class Precision; } }

namespace lucius
{

namespace ir
{

/*! \brief A class for representing a matrix value a program. */
class TensorValue : public Value
{
public:
    using Precision = matrix::Precision;

public:
    explicit TensorValue(std::shared_ptr<ValueImplementation>);
    explicit TensorValue(const Shape& t, const Precision& p);

public:
    const Shape& getShape() const;
};

} // namespace ir
} // namespace lucius







