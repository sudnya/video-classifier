/*    \file   LineSearchFactory.h
    \date   Saturday August 10, 2014
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The header file for the LineSearch class.
*/

#pragma once

// Standard Library Includes
#include <string>
#include <memory>

// Forward Declarations
namespace lucius { namespace optimizer { class LineSearch; } }

namespace lucius
{

namespace optimizer
{

/*! \brief A factory for line search algorithms */
class LineSearchFactory
{
public:
    static std::unique_ptr<LineSearch> create(const std::string& searchName);

public:
    static std::unique_ptr<LineSearch> create();

};

}

}

