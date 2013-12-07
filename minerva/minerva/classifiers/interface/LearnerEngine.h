/*	\file   LearnerEngine.h
	\date   Sunday August 11, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the LearnerEngine class.
*/

#pragma once

// Minerva Includes
#include <minerva/classifiers/interface/ClassifierEngine.h>

// Forward Declarations
namespace minerva { namespace classifiers { class Learner; } }

namespace minerva
{

namespace classifiers
{

class LearnerEngine : public ClassifierEngine
{
public:
	LearnerEngine();
	virtual ~LearnerEngine();

public:
	LearnerEngine(const LearnerEngine&) = delete;
	LearnerEngine& operator=(const LearnerEngine&) = delete;
	
private:
	virtual void registerModel();
	virtual void closeModel();

private:
	virtual void runOnImageBatch(const ImageVector& images);
	virtual size_t getInputFeatureCount() const;
	
	virtual bool requiresLabeledData() const;

private:
	Learner*     _learner;


};

}

}




