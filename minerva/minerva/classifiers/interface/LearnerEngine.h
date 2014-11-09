/*	\file   LearnerEngine.h
	\date   Sunday August 11, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the LearnerEngine class.
*/

#pragma once

// Minerva Includes
#include <minerva/classifiers/interface/Engine.h>

// Forward Declarations
namespace minerva { namespace classifiers { class Learner; } }

namespace minerva
{

namespace classifiers
{

class LearnerEngine : public Engine
{
public:
	LearnerEngine();
	virtual ~LearnerEngine();

public:
	LearnerEngine(const LearnerEngine&) = delete;
	LearnerEngine& operator=(const LearnerEngine&) = delete;
	
private:
	virtual void closeModel();

private:
	virtual ResultVector runOnBatch(Matrix&& input, Matrix&& reference);
	
	virtual bool requiresLabeledData() const;


};

}

}




