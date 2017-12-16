/*  \file   ConstantTensor.cpp
    \author Gregory Diamos
    \date   October 9, 2017
    \brief  The source file for the ConstantTensor class.
*/

// Lucius Includes
#include <lucius/ir/values/interface/ConstantTensor.h>

#include <lucius/ir/implementation/ConstantImplementation.h>

#include <lucius/ir/interface/Use.h>

#include <lucius/matrix/interface/Matrix.h>

namespace lucius
{

namespace ir
{

class ConstantTensorImplementation : public ConstantImplementation
{
public:
    ConstantTensorImplementation(const matrix::Matrix& m)
    : _matrix(m)
    {

    }

    virtual ~ConstantTensorImplementation()
    {

    }

public:
    matrix::Matrix& getContents()
    {
        return _matrix;
    }

    const matrix::Matrix& getContents() const
    {
        return _matrix;
    }

public:
    virtual std::shared_ptr<ValueImplementation> clone() const
    {
        return std::make_shared<ConstantTensorImplementation>(*this);
    }

public:
    std::string toString() const
    {
        return "Tensor(" + _matrix.shapeString() + ") constant";
    }

public:
    Type getType() const
    {
        return Type(Type::TensorId);
    }

private:
    matrix::Matrix _matrix;

};

ConstantTensor::ConstantTensor(const matrix::Matrix& value)
: Constant(std::make_shared<ConstantTensorImplementation>(value))
{

}

ConstantTensor::ConstantTensor(std::shared_ptr<ValueImplementation> implementation)
: Constant(implementation)
{

}

matrix::Matrix& ConstantTensor::getContents()
{
    return std::static_pointer_cast<ConstantTensorImplementation>(
        getValueImplementation())->getContents();
}

const matrix::Matrix& ConstantTensor::getContents() const
{
    return std::static_pointer_cast<ConstantTensorImplementation>(
        getValueImplementation())->getContents();
}

} // namespace ir
} // namespace lucius







