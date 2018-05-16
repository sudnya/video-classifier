/*  \file   FloatValue.h
    \author Gregory Diamos
    \date   October 11, 2017
    \brief  The header file for the FloatValue class.
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

/*! \brief A class for representing a single precision floating point value in a program for a
           generic machine.
*/
class FloatValue : public ir::TargetValue
{
public:
    using Context = ir::Context;

public:
    explicit FloatValue(std::shared_ptr<ir::ValueImplementation>);
    explicit FloatValue(Context& context);
};

} // namespace generic
} // namespace machine
} // namespace lucius










