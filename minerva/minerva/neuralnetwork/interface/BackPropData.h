/* Author: Sudnya Padalikar
 * Date  : 08/17/2013
 * The interface of the back propagate data class
 */

#pragma once

#include <vector>

#include <minerva/neuralnetwork/interface/Layer.h>
#include <minerva/matrix/interface/Matrix.h>
#include <minerva/neuralnetwork/interface/NeuralNetwork.h>

namespace minerva
{
namespace neuralnetwork
{
class BackPropData
{
    public:
        typedef minerva::neuralnetwork::NeuralNetwork NeuralNetwork;
        typedef minerva::matrix::Matrix Matrix;
        typedef minerva::matrix::BlockSparseMatrix BlockSparseMatrix;
        typedef minerva::matrix::Matrix::FloatVector FloatVector;
        typedef std::vector<BlockSparseMatrix> MatrixVector;
    public:
        BackPropData(NeuralNetwork* ann, const BlockSparseMatrix& input, const BlockSparseMatrix& ref);
        	
    public:
        MatrixVector getCostDerivative() const;
        NeuralNetwork* getNeuralNetworkPtr();

    public:
        Matrix getFlattenedWeights() const;
        Matrix getFlattenedCostDerivative() const;
        void   setFlattenedWeights(const Matrix& weights);
        float  computeCostForNewFlattenedWeights(const Matrix& weights) const;
        float  computeAccuracyForNewFlattenedWeights(const Matrix& weights) const;
        Matrix computePartialDerivativesForNewFlattenedWeights(const Matrix& weights) const;
		
    private:
        bool testDerivative();

    private:
        MatrixVector getDeltas(const NeuralNetwork& network, const MatrixVector& m) const;
        MatrixVector getActivations(const NeuralNetwork& network) const;
        Matrix sigmoidDerivative(const Matrix& m) const;
       
    private:
        NeuralNetwork createNetworkFromWeights(const Matrix& weights) const;
        MatrixVector getCostDerivative(const NeuralNetwork& network) const;

    private:
        NeuralNetwork* m_neuralNetworkPtr;
        BlockSparseMatrix m_input;
        BlockSparseMatrix m_referenceOutput;

	private:
		float m_lambda; // cost function regularization

};

}
}

