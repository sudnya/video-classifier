/*    \file   ModelSpecification.h
    \date   Saturday April 26, 2014
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The header file for the ModelSpecification class.
*/

#pragma once

// Standard Library Includes
#include <string>
#include <memory>

// Forward Declarations
namespace lucius { namespace model { class Model;                            } }
namespace lucius { namespace model { class ModelSpecificationImplementation; } }

namespace lucius
{

namespace model
{

/*! \brief A class for initializing a classificaiton model
    from a json specification */
class ModelSpecification 
{
public:
    ModelSpecification(const std::string& specification = "");
    ~ModelSpecification();

public:
    void parseSpecification(const std::string& specification);
    void initializeModel(Model& model);

private:
    std::unique_ptr<ModelSpecificationImplementation> _implementation;

};

}

}

