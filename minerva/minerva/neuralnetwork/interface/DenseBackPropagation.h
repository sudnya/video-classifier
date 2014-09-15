/* Author: Sudnya Padalikar
 * Date  : 08/17/2013
 * The interface of the back propagate data class
 */

#pragma once

#include <minerva/neuralnetwork/interface/BackPropagation.h>

namespace minerva
{

namespace neuralnetwork
{

class DenseBackPropagation : public BackPropagation
{
public:
	DenseBackPropagation(NeuralNetwork* ann = nullptr,
		BlockSparseMatrix* input = nullptr,
		BlockSparseMatrix* ref = nullptr);

private:
	virtual BlockSparseMatrixVector getCostDerivative(const NeuralNetwork&, const BlockSparseMatrix&, const BlockSparseMatrix& ) const;
	virtual BlockSparseMatrix getInputDerivative(const NeuralNetwork&, const BlockSparseMatrix&, const BlockSparseMatrix&) const;
	virtual float getCost(const NeuralNetwork&, const BlockSparseMatrix&, const BlockSparseMatrix&) const;
	virtual float getInputCost(const NeuralNetwork&, const BlockSparseMatrix&, const BlockSparseMatrix&) const;

private:
	bool testDerivative();

private:
	BlockSparseMatrix getInputDelta(const NeuralNetwork& network, const BlockSparseMatrixVector& m) const;
	BlockSparseMatrixVector getDeltas(const NeuralNetwork& network, const BlockSparseMatrixVector& m) const;
	BlockSparseMatrixVector getActivations(const NeuralNetwork& network, const BlockSparseMatrix& inputs) const;
   
private:
	BlockSparseMatrixVector getCostDerivative(const NeuralNetwork& network) const;
	BlockSparseMatrix getInputDerivative(const NeuralNetwork& network,
		const BlockSparseMatrix& input) const;

private:
	float _lambda; // cost function regularization

};

}

}


