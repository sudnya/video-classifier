/*  \file   TensorData.cpp
    \author Gregory Diamos
    \date   October 1, 2017
    \brief  The source file for the TensorData class.
*/

// Lucius Includes
#include <lucius/ir/target/interface/TensorData.h>

#include <lucius/ir/target/implementation/TargetValueDataImplementation.h>

#include <lucius/matrix/interface/Matrix.h>

namespace lucius
{

namespace ir
{

class TensorDataImplementation : public TargetValueDataImplementation
{
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

TensorData::TensorData()
: TensorData(std::make_shared<TensorDataImplementation>())
{

}

TensorData::TensorData(std::shared_ptr<TargetValueDataImplementation> implementation)
: _implementation(std::static_pointer_cast<TensorDataImplementation>(implementation))
{
    assert(static_cast<bool>(std::dynamic_pointer_cast<TensorDataImplementation>(implementation)));
}

matrix::Matrix TensorData::getTensor() const
{
    return _implementation->getTensor();
}

std::shared_ptr<TargetValueDataImplementation> TensorData::getImplementation() const
{
    return _implementation;
}

} // namespace ir
} // namespace lucius









