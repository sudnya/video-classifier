/*  \file   PassFactory.h
    \author Gregory Diamos
    \date   May 4, 2017
    \brief  The header file for the PassFactory class.
*/

#pragma once

// forward declarations
namespace lucius { namespace ir { class Function; } }

namespace lucius
{
namespace optimization
{

/*! \brief A class used to construct passes. */
class PassFactory
{
public:
    static std::unique_ptr<Pass> create(const std::string& );
};

} // namespace optimization
} // namespace lucius






