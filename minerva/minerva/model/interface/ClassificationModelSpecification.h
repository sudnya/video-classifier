/*	\file   ClassificationModelSpecification.h
	\date   Saturday April 26, 2014
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the ClassificationModelSpecification class.
*/

#pragma once

// Standard Library Includes
#include <string>
#include <memory>

// Forward Declarations
namespace minerva { namespace model { class ClassificationModel;                            } }
namespace minerva { namespace model { class ClassificationModelSpecificationImplementation; } }

namespace minerva
{

namespace model
{

/*! \brief A class for initializing a classificaiton model
	from a json specification */
class ClassificationModelSpecification 
{
public:
	ClassificationModelSpecification();
	~ClassificationModelSpecification();

public:
	void parseSpecification(const std::string& specification);
	void initializeModel(ClassificationModel* model);

private:
	std::unique_ptr<ClassificationModelSpecificationImplementation> _implementation;

};

}

}

