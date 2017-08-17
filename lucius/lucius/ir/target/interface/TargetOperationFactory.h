/*  \file   TargetOperationFactory.h
    \author Gregory Diamos
    \date   August 16, 2017
    \brief  The header file for the TargetOperationFactory class.
*/

#pragma once

// Standard Library Includes
#include <string>

// Forward Declarations
namespace lucius { namespace ir { class TargetOperation; } }

namespace lucius
{

namespace ir
{

/*! \brief A factory for creating target operations. */
class TargetOperationFactory
{
public:
    static TargetOperation create(const std::string& name);
};

} // namespace ir
} // namespace lucius


