/*  \file   RandomStateData.cpp
    \author Gregory Diamos
    \date   October 1, 2017
    \brief  The source file for the RandomStateData class.
*/

// Lucius Includes
#include <lucius/machine/generic/interface/RandomStateData.h>

#include <lucius/ir/target/implementation/TargetValueDataImplementation.h>

#include <lucius/matrix/interface/RandomOperations.h>

namespace lucius
{
namespace machine
{
namespace generic
{

class RandomStateDataImplementation : public ir::TargetValueDataImplementation
{
public:
    RandomStateDataImplementation()
    {
        // intentionally blank
    }

public:
    matrix::RandomState& getRandomState() const
    {
        return const_cast<matrix::RandomState&>(_state);
    }

public:
    virtual void* getData() const
    {
        return const_cast<void*>(reinterpret_cast<const void*>(&_state));
    }

private:
    matrix::RandomState _state;

};

RandomStateData::RandomStateData()
: RandomStateData(std::make_shared<RandomStateDataImplementation>())
{

}

RandomStateData::RandomStateData(std::shared_ptr<ir::TargetValueDataImplementation> implementation)
: _implementation(std::static_pointer_cast<RandomStateDataImplementation>(implementation))
{
    assert(static_cast<bool>(std::dynamic_pointer_cast<RandomStateDataImplementation>(
        implementation)));
}

matrix::RandomState& RandomStateData::getRandomState() const
{
    return _implementation->getRandomState();
}

std::shared_ptr<ir::TargetValueDataImplementation> RandomStateData::getImplementation() const
{
    return _implementation;
}

} // namespace generic
} // namespace machine
} // namespace lucius










