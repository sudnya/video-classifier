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
	virtual MatrixVector getCostDerivative(const NeuralNetwork&, const BlockSparseMatrix&, const BlockSparseMatrix& ) const;
	virtual BlockSparseMatrix getInputDerivative(const NeuralNetwork&, const BlockSparseMatrix&, const BlockSparseMatrix&) const;
	virtual float getCost(const NeuralNetwork&, const BlockSparseMatrix&, const BlockSparseMatrix&) const;

private:
	bool testDerivative();

private:
	BlockSparseMatrix getInputDelta(const NeuralNetwork& network, const MatrixVector& m) const;
	MatrixVector getDeltas(const NeuralNetwork& network, const MatrixVector& m) const;
	MatrixVector getActivations(const NeuralNetwork& network, const BlockSparseMatrix& inputs) const;
   
private:
	MatrixVector getCostDerivative(const NeuralNetwork& network) const;
	BlockSparseMatrix getInputDerivative(const NeuralNetwork& network,
		const BlockSparseMatrix& input) const;

private:
	float _lambda; // cost function regularization

};

}

}


