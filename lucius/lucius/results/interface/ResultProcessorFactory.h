/*! \file   ResultProcessorFactory.h
    \date   Sunday January 11, 2015
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The header file for the ResultProcessorfactor class.
*/

#pragma once

// Standard Library Includes
#include <string>

// Forward Declarations
namespace lucius { namespace results { class ResultProcessor; } }

namespace lucius
{

namespace results
{


/*! \brief A factory for result processors. */
class ResultProcessorFactory
{
public:
    /*! \brief Create a new instance of the named processor. */
    static ResultProcessor* create(const std::string& );
    
    /*! \brief Create a new instance of the default processor. */
    static ResultProcessor* create();

};

}

}


