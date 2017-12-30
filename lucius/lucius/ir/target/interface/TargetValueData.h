/*  \file   TargetValueData.h
    \author Gregory Diamos
    \date   October 1, 2017
    \brief  The header file for the TargetValueData class.
*/

#pragma once

// Standard Library Includes
#include <memory>

// Forward Declarations
namespace lucius { namespace ir { class TargetValueDataImplementation; } }

namespace lucius
{

namespace ir
{

/*! \brief A class for representing a value resource. */
class TargetValueData
{
public:
    void* data() const;

private:
    std::shared_ptr<TargetValueDataImplementation> _implementation;
};

} // namespace ir
} // namespace lucius







