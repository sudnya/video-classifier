/*  \file   DataFactory.h
    \author Gregory Diamos
    \date   April 24, 2018
    \brief  The header file for the DataFactory class.
*/

#pragma once

// Forward Declarations
namespace lucius { namespace ir { class TargetValueData; } }
namespace lucius { namespace ir { class Type;            } }

namespace lucius
{
namespace machine
{
namespace generic
{

/*! \brief A factory for creating generic data based on their type. */
class DataFactory
{
public:
    static ir::TargetValueData create(const ir::Type& type);
};

} // namespace generic
} // namespace machine
} // namespace lucius


