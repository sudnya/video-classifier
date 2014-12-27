/*	\file   LearnerEngine.cpp
	\date   Sunday August 11, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the LearnerEngine class.
*/

// Minerva Includes
#include <minerva/classifiers/interface/LearnerEngine.h>

#include <minerva/network/interface/NeuralNetwork.h>
#include <minerva/network/interface/Layer.h>
#include <minerva/network/interface/CostFunction.h>

#include <minerva/optimizer/interface/NeuralNetworkSolver.h>

#include <minerva/results/interface/ResultVector.h>

#include <minerva/matrix/interface/Matrix.h>

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
	
	network.train(std::move(input), std::move(reference));
	
	restoreAggregateNetwork(network);
	
	return ResultVector();
}

bool LearnerEngine::requiresLabeledData() const
{
	return true;
}

}

}




