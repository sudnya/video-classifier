/*	\file   WeightRegularizationCostFunction.cpp
	\date   November 19, 2014
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the WeightRegularizationCostFunction class.
*/

// Minerva Includes
#include <minerva/network/interface/WeightRegularizationCostFunction.h>
#include <minerva/matrix/interface/BlockSparseMatrix.h>

#include <minerva/util/interface/Knobs.h>

namespace minerva
{

namespace network
{

WeightRegularizationCostFunction::~WeightRegularizationCostFunction()
{

}

float WeightRegularizationCostFunction::getCost(const BlockSparseMatrix& weights) const
{
	float lambda = util::KnobDatabase::getKnobValue("NeuralNetwork::Lambda", 0.001);

	return weights.elementMultiply(weights).reduceSum() * (lambda / 2.0f);
}

WeightRegularizationCostFunction::BlockSparseMatrix WeightRegularizationCostFunction::getGradient(const BlockSparseMatrix& weights) const
{
	float lambda = util::KnobDatabase::getKnobValue("NeuralNetwork::Lambda", 0.001);
	
	return weights.multiply(lambda);
}

WeightCostFunction* WeightRegularizationCostFunction::clone() const
{
	return new WeightRegularizationCostFunction;
}

}

}



