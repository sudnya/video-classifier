/*	\file   LearnerEngine.h
	\date   Sunday August 11, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the LearnerEngine class.
*/

#pragma once

// Minerva Includes
#include <minerva/classifiers/interface/ClassifierEngine.h>

namespace minerva
{

namespace classifiers
{

class LearnerEngine : public ClassifierEngine
{
public:
	LearnerEngine();
	
protected:
	virtual void runOnImageBatch(const ImageVector& images);
	virtual size_t getInputFeatureCount() const;

};

}

}




