/*  \file   IntegerData.h
    \author Gregory Diamos
    \date   October 1, 2017
    \brief  The header file for the IntegerData class.
*/

#pragma once

// Standard Library Includes
#include <memory>

// Forward Declarations
namespace lucius { namespace ir { class TargetValueDataImplementation; } }

namespace lucius { namespace machine { namespace generic { class IntegerDataImplementation; } } }

namespace lucius
{
namespace machine
{
namespace generic
{


/*! \brief A class for representing a tensor value resource. */
class IntegerData
{
public:
    IntegerData();
    explicit IntegerData(size_t value);
    IntegerData(std::shared_ptr<ir::TargetValueDataImplementation> implementation);

public:
    size_t getInteger() const;
    void setInteger(size_t value);

public:
    std::shared_ptr<ir::TargetValueDataImplementation> getImplementation() const;

private:
    std::shared_ptr<IntegerDataImplementation> _implementation;

};

} // namespace generic
} // namespace machine
} // namespace lucius










