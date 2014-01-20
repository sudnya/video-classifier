/*! \file   FeatureExtractor.h
	\date   Saturday January 18, 2014
	\author Gregory Diamos <gregory.diamos@gmail.com>
	\brief  The header file for the FeatureExtractor class.
*/

#pragma once

// Forward Declarations
namespace minerva { namespace model  { class ClassificationModel; } }
namespace minerva { namespace matrix { class Matrix;              } }
namespace minerva { namespace video  { class ImageVector;         } }

// Standard Library Includes
#include <cstddef>

namespace minerva
{

namespace classifiers
{

/*! \brief A class for extracting features from raw input */
class FeatureExtractor
{
public:
	typedef model::ClassificationModel ClassificationModel;
	typedef matrix::Matrix             Matrix;
	typedef video::ImageVector         ImageVector;

public:
	FeatureExtractor(ClassificationModel* model);

public:
	size_t getInputFeatureCount() const;
	Matrix extract(ImageVector&& images);

private:
	ClassificationModel* _classificationModel;

};

}

}



