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
namespace lucius { namespace ir { class IntegerDataImplementation;    } }

namespace lucius
{

namespace ir
{

/*! \brief A class for representing a tensor value resource. */
class IntegerData
{
public:
    IntegerData();
    IntegerData(size_t value);
    IntegerData(std::shared_ptr<TargetValueDataImplementation> implementation);

public:
    size_t getInteger() const;

public:
    std::shared_ptr<TargetValueDataImplementation> getImplementation() const;

private:
    std::shared_ptr<IntegerDataImplementation> _implementation;

};

} // namespace ir
} // namespace lucius










