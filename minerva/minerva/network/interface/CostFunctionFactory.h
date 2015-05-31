/*    \file   CostFunctionFactory.h
    \date   December 25, 2014
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The header file for the CostFunctionFactory class.
*/

#pragma once

// Standard Library Includes
#include <string>

// Forward Declarations
namespace minerva { namespace network { class CostFunction; } }

namespace minerva
{

namespace network
{

/*! \brief A factory for cost functions. */
class CostFunctionFactory
{
public:
    static CostFunction* create(const std::string& costFunctionName);

    // Create the default cost function
    static CostFunction* create();

};

}

}


