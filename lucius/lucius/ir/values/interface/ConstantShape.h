/*  \file   ConstantShape.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the ConstantShape class.
*/

#pragma once

// Lucius Includes
#include <lucius/ir/interface/Constant.h>

// Forward Declarations
namespace lucius { namespace matrix { class Dimension; } }
namespace lucius { namespace ir     { class Shape;     } }

namespace lucius
{

namespace ir
{

/*! \brief A class for representing a constant shape value in a program. */
class ConstantShape : public Constant
{
public:
    explicit ConstantShape(const matrix::Dimension& value);
    explicit ConstantShape(std::shared_ptr<ValueImplementation>);

public:
    Shape& getContents();
    const Shape& getContents() const;


};

} // namespace ir
} // namespace lucius







