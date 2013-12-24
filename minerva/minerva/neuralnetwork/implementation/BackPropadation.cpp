/*! \file   BackPropagation.cpp
	\author Gregory Diamos
	\date   Sunday December 22, 2013
	\brief  The source file for the BackPropagation class.
*/

// Minerva Includes
#include <minerva/neuralnetwork/interface/BackPropagation.h>
#include <minerva/neuralnetwork/interface/NeuralNetwork.h>

#include <minerva/matrix/interface/Matrix.h>
#include <minerva/matrix/interface/BlockSparseMatrix.h>

namespace minerva
{
namespace neuralnetwork
{

typedef matrix::Matrix Matrix;
typedef matrix::BlockSparseMatrix BlockSparseMatrix;
typedef Matrix::FloatVector FloatVector;
typedef BackPropagation::MatrixVector MatrixVector;

BackPropagation::BackPropagation(NeuralNetwork* ann,
	BlockSparseMatrix* input,
	BlockSparseMatrix* ref)
: _neuralNetworkPtr(ann), _input(input), _referenceOutput(ref)
{

}

BackPropagation::~BackPropagation()
{

}

BackPropagation::MatrixVector BackPropagation::computeCostDerivative() const
{
	return getCostDerivative(*getNeuralNetwork(), *getInput(), *getReferenceOutput());
}

BackPropagation::BlockSparseMatrix BackPropagation::computeInputDerivative() const
{
	return getInputDerivative(*getNeuralNetwork(), *getInput(), *getReferenceOutput());
}

float BackPropagation::computeCost() const
{
	return getCost(*getNeuralNetwork(), *getInput(), *getReferenceOutput());
}

BackPropagation::NeuralNetwork* BackPropagation::getNeuralNetwork()
{
	return _neuralNetworkPtr;
}

BackPropagation::BlockSparseMatrix* BackPropagation::getInput()
{
	return _input;
}

BackPropagation::BlockSparseMatrix* BackPropagation::getReferenceOutput()
{
	return _referenceOutput;
}
	
const BackPropagation::NeuralNetwork* BackPropagation::getNeuralNetwork() const
{
	return _neuralNetworkPtr;
}

const BackPropagation::BlockSparseMatrix* BackPropagation::getInput() const
{
	return _input;
}

const BackPropagation::BlockSparseMatrix* BackPropagation::getReferenceOutput() const
{
	return _referenceOutput;
}

void BackPropagation::setNeuralNetwork(NeuralNetwork* n)
{
	_neuralNetworkPtr = n;
}

void BackPropagation::setInput(BlockSparseMatrix* i)
{
	_input = i;
}

void BackPropagation::setReferenceOutput(BlockSparseMatrix* o)
{
	_referenceOutput = o;
}

static size_t getElementCount(const MatrixVector& matrices)
{
	size_t size = 0;
	
	for(auto& matrix : matrices)
	{
		size += matrix.size();
	}
	
	return size;
}

Matrix BackPropagation::flatten(const MatrixVector& matrices)
{
	FloatVector flattenedData(getElementCount(matrices));
	
	size_t position = 0;
	
	for(auto& blockedMatrix : matrices)
	{
		for(auto& matrix : blockedMatrix)
		{
			auto data = matrix.data();
			
			std::memcpy(&flattenedData[position], data.data(),
				sizeof(float) * matrix.size());
			
			position += matrix.size();
		}
	}

	return Matrix(1, getElementCount(matrices), flattenedData);
}

Matrix BackPropagation::flatten(const BlockSparseMatrix& blockedMatrix)
{
	FloatVector flattenedData(blockedMatrix.size());
	
	size_t position = 0;
	
	for(auto& matrix : blockedMatrix)
	{
		auto data = matrix.data();
		
		std::memcpy(&flattenedData[position], data.data(),
			sizeof(float) * matrix.size());
		
		position += matrix.size();
	}

	return Matrix(1, blockedMatrix.size(), flattenedData);
} 

Matrix BackPropagation::getFlattenedWeights() const
{
	return getNeuralNetwork()->getFlattenedWeights();
}

Matrix BackPropagation::getFlattenedCostDerivative() const
{
	return flatten(computeCostDerivative());
}

static NeuralNetwork createNetworkFromWeights(
	const NeuralNetwork* neuralNetwork, const Matrix& weights)
{
	NeuralNetwork newNetwork(*neuralNetwork);
	
	newNetwork.setFlattenedWeights(weights);
		
	return newNetwork;
}

void BackPropagation::setFlattenedWeights(const Matrix& weights)
{
	*_neuralNetworkPtr = createNetworkFromWeights(getNeuralNetwork(), weights);
}

float BackPropagation::computeCostForNewFlattenedWeights(const Matrix& weights) const
{
	auto network = createNetworkFromWeights(getNeuralNetwork(), weights);

	return getCost(network, *getInput(), *getReferenceOutput());
}

float BackPropagation::computeCostForNewFlattenedInputs(const Matrix& inputs) const
{
	return getCost(*getNeuralNetwork(),
		getNeuralNetwork()->convertToBlockSparseForLayerInput(getNeuralNetwork()->front(), inputs),
		*getReferenceOutput());
}

float BackPropagation::computeAccuracyForNewFlattenedWeights(const Matrix& weights) const
{
	auto network = createNetworkFromWeights(getNeuralNetwork(), weights);

	return network.computeAccuracy(*getInput(), *getReferenceOutput());
}

Matrix BackPropagation::computePartialDerivativesForNewFlattenedWeights(const Matrix& weights) const
{
	auto network = createNetworkFromWeights(getNeuralNetwork(), weights);

	return flatten(getCostDerivative(network, *getInput(), *getReferenceOutput()));
}

Matrix BackPropagation::computePartialDerivativesForNewFlattenedInputs(const Matrix& inputs) const
{
	return flatten(getInputDerivative(*getNeuralNetwork(),
		getNeuralNetwork()->convertToBlockSparseForLayerInput(getNeuralNetwork()->front(),
		inputs), *getReferenceOutput()));
}

}

}

