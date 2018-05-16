/*  \file   RandomStateData.h
    \author Gregory Diamos
    \date   October 1, 2017
    \brief  The header file for the RandomStateData class.
*/

#pragma once

// Standard Library Includes
#include <memory>

// Forward Declarations
namespace lucius { namespace ir { class TargetValueDataImplementation; } }

namespace lucius { namespace machine { namespace generic { class RandomStateDataImplementation; } } }

namespace lucius { namespace matrix { class RandomState; } }

namespace lucius
{
namespace machine
{
namespace generic
{

/*! \brief A class for representing a random state resource. */
class RandomStateData
{
public:
    RandomStateData();
    RandomStateData(std::shared_ptr<ir::TargetValueDataImplementation> implementation);

public:
    matrix::RandomState& getRandomState() const;

public:
    std::shared_ptr<ir::TargetValueDataImplementation> getImplementation() const;

private:
    std::shared_ptr<RandomStateDataImplementation> _implementation;

};

} // namespace generic
} // namespace machine
} // namespace lucius









