/*  \file   TensorType.cpp
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The source file for the TensorType class.
*/

// Lucius Includes
#include <lucius/ir/types/interface/TensorType.h>

#include <lucius/matrix/interface/Precision.h>

#include <lucius/ir/interface/Shape.h>

#include <lucius/ir/implementation/TypeImplementation.h>


namespace lucius
{

namespace ir
{

using Precision = matrix::Precision;

class TensorTypeImplementation : public TypeImplementation
{
public:
    TensorTypeImplementation(const Shape& shape, const Precision& precision)
    : _shape(shape), _precision(precision)
    {

    }

    const Shape& getShape() const
    {
        return _shape;
    }

    const Precision& getPrecision() const
    {
        return _precision;
    }

private:
    Shape     _shape;
    Precision _precision;

};

TensorType::TensorType(std::shared_ptr<TypeImplementation> implementation)
: Type(implementation)
{

}

TensorType::TensorType(const Shape& d, const Precision& p)
: Type(std::make_shared<TensorTypeImplementation>(d, p))
{

}

TensorType::~TensorType()
{

}

const Shape& TensorType::getShape() const
{
    auto implementation = std::static_pointer_cast<TensorTypeImplementation>(
        getTypeImplementation());

    return implementation->getShape();
}

} // namespace ir
} // namespace lucius




