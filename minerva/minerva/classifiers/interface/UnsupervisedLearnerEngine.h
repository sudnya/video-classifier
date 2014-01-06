/*	\file   UnsupervisedLearnerEngine.h
	\date   Sunday August 11, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the UnsupervisedLearnerEngine class.
*/

#pragma once

// Minerva Includes
#include <minerva/classifiers/interface/ClassifierEngine.h>

// Forward Declaration
namespace minerva { namespace classifiers { class UnsupervisedLearner; } }

namespace minerva
{

namespace classifiers
{

class UnsupervisedLearnerEngine : public ClassifierEngine
{
public:
	UnsupervisedLearnerEngine();
	virtual ~UnsupervisedLearnerEngine();

public:
	UnsupervisedLearnerEngine(const UnsupervisedLearnerEngine&) = delete;
	UnsupervisedLearnerEngine& operator=(const UnsupervisedLearnerEngine&) = delete;
	
private:
	virtual void registerModel();
	virtual void closeModel();
	
private:
	virtual void runOnImageBatch(ImageVector&& images);
	virtual size_t getInputFeatureCount() const;

private:
	UnsupervisedLearner* _learner;

};

}

}




