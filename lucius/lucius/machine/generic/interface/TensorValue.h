/*  \file   TensorValue.h
    \author Gregory Diamos
    \date   October 11, 2017
    \brief  The header file for the TensorValue class.
*/

#pragma once

// Lucius Includes
#include <lucius/ir/target/interface/TargetValue.h>

// Forward Declarations
namespace lucius { namespace ir { class Shape;   } }
namespace lucius { namespace ir { class Context; } }

namespace lucius { namespace matrix { class Precision; } }

namespace lucius
{
namespace machine
{
namespace generic
{

/*! \brief A class for representing a tensor value in a program for a generic machine. */
class TensorValue : public ir::TargetValue
{
public:
    using Precision = matrix::Precision;
    using Shape = ir::Shape;
    using Context = ir::Context;

public:
    explicit TensorValue(std::shared_ptr<ir::ValueImplementation>);
    explicit TensorValue(const Shape& t, const Precision& p, Context& context);

public:
    const Shape& getShape() const;
    const Precision& getPrecision() const;
};

} // namespace generic
} // namespace machine
} // namespace lucius








