/*  \file   RandomStateValue.h
    \author Gregory Diamos
    \date   October 11, 2017
    \brief  The header file for the RandomStateValue class.
*/

#pragma once

// Lucius Includes
#include <lucius/ir/target/interface/TargetValue.h>

// Forward Declarations
namespace lucius { namespace ir { class Context; } }

namespace lucius
{
namespace machine
{
namespace generic
{

/*! \brief A class for representing a random state value in a program for a generic machine. */
class RandomStateValue : public ir::TargetValue
{
public:
    using Context = ir::Context;

public:
    explicit RandomStateValue(std::shared_ptr<ir::ValueImplementation>);
    explicit RandomStateValue(Context& context);
};

} // namespace generic
} // namespace machine
} // namespace lucius









