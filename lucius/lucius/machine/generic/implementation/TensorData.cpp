/*  \file   TensorData.cpp
    \author Gregory Diamos
    \date   October 1, 2017
    \brief  The source file for the TensorData class.
*/

// Lucius Includes
#include <lucius/machine/generic/interface/TensorData.h>

#include <lucius/ir/interface/Shape.h>

#include <lucius/ir/target/implementation/TargetValueDataImplementation.h>

#include <lucius/matrix/interface/Matrix.h>

namespace lucius
{
namespace machine
{
namespace generic
{

class TensorDataImplementation : public ir::TargetValueDataImplementation
{
public:
    TensorDataImplementation(const ir::Shape& shape, const matrix::Precision& precision)
    : _tensor(shape.getDimension(), precision)
    {

    }

    TensorDataImplementation()
    {
        // intentionally blank
    }

public:
    matrix::Matrix getTensor() const
    {
        return _tensor;
    }

public:
    virtual void* getData() const
    {
        return const_cast<void*>(reinterpret_cast<const void*>(&_tensor));
    }

private:
    matrix::Matrix _tensor;

};

TensorData::TensorData(const ir::Shape& shape, const matrix::Precision& precision)
: TensorData(std::make_shared<TensorDataImplementation>(shape, precision))
{

}

TensorData::TensorData()
: TensorData(std::make_shared<TensorDataImplementation>())
{

}

TensorData::TensorData(std::shared_ptr<ir::TargetValueDataImplementation> implementation)
: _implementation(std::static_pointer_cast<TensorDataImplementation>(implementation))
{
    assert(static_cast<bool>(std::dynamic_pointer_cast<TensorDataImplementation>(implementation)));
}

matrix::Matrix TensorData::getTensor() const
{
    return _implementation->getTensor();
}

std::shared_ptr<ir::TargetValueDataImplementation> TensorData::getImplementation() const
{
    return _implementation;
}

} // namespace generic
} // namespace machine
} // namespace lucius








