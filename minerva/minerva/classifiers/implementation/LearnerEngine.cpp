/*	\file   LearnerEngine.cpp
	\date   Sunday August 11, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the LearnerEngine class.
*/

// Minerva Includes
#include <minerva/classifiers/interface/LearnerEngine.h>
#include <minerva/classifiers/interface/Learner.h>

#include <minerva/util/interface/debug.h>

// Standard Library Includes
#include <cassert>

namespace minerva
{

namespace classifiers
{

LearnerEngine::LearnerEngine()
{
	
}

LearnerEngine::~LearnerEngine()
{

}

void LearnerEngine::closeModel()
{
	saveModel();
}
	
LearnerEngine::ResultVector LearnerEngine::runOnBatch(Matrix&& input, Matrix&& reference)
{
	util::log("LearnerEngine") << "Performing supervised "
		"learning on batch of " << input.rows() <<  " images...\n";
	
	auto network = getAggregateNetwork();
	
	network.train(input, reference);
	
	restoreAggregateNetwork(network);
	
	return ResultVector();
}

bool LearnerEngine::requiresLabeledData() const
{
	return true;
}

}

}




