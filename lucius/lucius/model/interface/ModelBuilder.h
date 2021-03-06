/*    \file   ModelBuilder.h
    \date   Saturday August 10, 2013
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The header file for the ModelBuilder class.
*/

#pragma once

// Standard Library Includes
#include <string>
#include <memory>

// Forward Declarations
namespace lucius { namespace model { class Model; } }

namespace lucius
{

namespace model
{

/*! \brief A constructor for classification models */
class ModelBuilder
{
public:
    /*! \brief Create a new model using the specified topology. */
    static std::unique_ptr<Model> create(const std::string& specificationString);

};

}

}


