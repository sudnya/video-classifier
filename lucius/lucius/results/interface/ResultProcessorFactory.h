/*! \file   ResultProcessorFactory.h
    \date   Sunday January 11, 2015
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The header file for the ResultProcessorfactor class.
*/

#pragma once

// Standard Library Includes
#include <string>
#include <memory>

// Forward Declarations
namespace lucius { namespace results { class ResultProcessor; } }
namespace lucius { namespace util    { class ParameterPack;   } }

namespace lucius
{

namespace results
{


/*! \brief A factory for result processors. */
class ResultProcessorFactory
{
public:
    /*! \brief Create a new instance of the named processor with parameters. */
    static std::unique_ptr<ResultProcessor>
        create(const std::string& , const util::ParameterPack& pack);

    /*! \brief Create a new instance of the named processor. */
    static std::unique_ptr<ResultProcessor> create(const std::string& );

    /*! \brief Create a new instance of the default processor. */
    static std::unique_ptr<ResultProcessor> create();

};

}

}


