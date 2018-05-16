/*  \file   PassFactory.h
    \author Gregory Diamos
    \date   May 4, 2017
    \brief  The header file for the PassFactory class.
*/

#pragma once

// Standard Library Includes
#include <string>

// Forward Declarations
namespace lucius { namespace ir { class Function; } }

namespace lucius { namespace optimization { class Pass; } }

namespace lucius { namespace util { class Any; } }

namespace lucius
{
namespace optimization
{

/*! \brief A class used to construct passes. */
class PassFactory
{
public:
    static std::unique_ptr<Pass> create(const std::string& );
    static std::unique_ptr<Pass> create(const std::string& , const util::Any& parameters);
};

} // namespace optimization
} // namespace lucius






