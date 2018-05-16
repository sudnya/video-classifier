/*  \file   StructureValue.h
    \author Gregory Diamos
    \date   October 11, 2017
    \brief  The header file for the StructureValue class.
*/

#pragma once

// Lucius Includes
#include <lucius/ir/target/interface/TargetValue.h>

// Forward Declarations
namespace lucius { namespace ir { class Context; } }
namespace lucius { namespace ir { class Type;    } }

namespace lucius
{
namespace machine
{
namespace generic
{

/*! \brief A class for representing a structure of values. */
class StructureValue : public ir::TargetValue
{
public:
    explicit StructureValue(std::shared_ptr<ir::ValueImplementation>);
    explicit StructureValue(const ir::Type& type, ir::Context& context);

};

} // namespace generic
} // namespace machine
} // namespace lucius



