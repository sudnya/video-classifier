/*  \file   FloatData.h
    \author Gregory Diamos
    \date   October 1, 2017
    \brief  The header file for the FloatData class.
*/

#pragma once

// Standard Library Includes
#include <memory>

// Forward Declarations
namespace lucius { namespace ir { class TargetValueDataImplementation; } }

namespace lucius { namespace machine { namespace generic { class FloatDataImplementation; } } }

namespace lucius
{
namespace machine
{
namespace generic
{

/*! \brief A class for representing a tensor value resource. */
class FloatData
{
public:
    FloatData();
    FloatData(float value);
    FloatData(std::shared_ptr<ir::TargetValueDataImplementation> implementation);

public:
    float getFloat() const;

public:
    std::shared_ptr<ir::TargetValueDataImplementation> getImplementation() const;

private:
    std::shared_ptr<FloatDataImplementation> _implementation;

};

} // namespace generic
} // namespace machine
} // namespace lucius











