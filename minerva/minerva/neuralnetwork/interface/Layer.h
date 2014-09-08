/* Author: Sudnya Padalikar
 * Date  : 08/11/2013
 * The interface for the Layer class 
 */

#pragma once

// Minerva Includes
#include <minerva/matrix/interface/BlockSparseMatrix.h>

// Standard Library Includes
#include <random>
#include <set>

namespace minerva
{
namespace neuralnetwork
{

/* \brief A neural network layer.

	Each layer is stored as a block sparse matrix.  Rows correspond to output neurons,
	and columns correspond to input weights.  
*/
class Layer
{
	public:
		typedef minerva::matrix::Matrix Matrix;
		typedef minerva::matrix::BlockSparseMatrix BlockSparseMatrix;
		
		typedef BlockSparseMatrix::iterator	   iterator;
		typedef BlockSparseMatrix::const_iterator const_iterator;
		
		typedef std::set<size_t> NeuronSet;

	public:
		Layer(unsigned totalBlocks = 0, size_t blockInput = 0,
			size_t blockOutput = 0, size_t blockStep = 0);

		Layer(const Layer&);
		Layer& operator=(const Layer&);
		Layer& operator=(Layer&&);

	public:
		void initializeRandomly(std::default_random_engine& engine, float epsilon = 6.0f);
		BlockSparseMatrix runInputs(const BlockSparseMatrix& m) const;
		BlockSparseMatrix runReverse(const BlockSparseMatrix& m) const;
 
 	public:
 		void transpose();
	
	public:
		size_t getInputCount()  const;
		size_t getOutputCount() const;

		size_t getInputBlockingFactor() const;
		size_t getOutputBlockingFactor() const;

	public:
		size_t getOutputCountForInputCount(size_t inputCount) const;

	public:
		size_t getFloatingPointOperationCount() const;
	
	public:
		size_t totalNeurons()	 const;
		size_t totalConnections() const;

		size_t blockSize() const;

	public:
		const BlockSparseMatrix& getWeightsWithoutBias() const;
		void setWeightsWithoutBias(const BlockSparseMatrix& );		

	public:
		size_t totalWeights() const;
		
	public:
		Matrix getFlattenedWeights() const;
		void setFlattenedWeights(const Matrix& m);

	public:
		void resize(size_t blocks);
		void resize(size_t blocks, size_t blockInput,
			size_t blockOutput);

	public:
		iterator	   begin();
		const_iterator begin() const;

		iterator	   end();
		const_iterator end() const;

	public:
		iterator	   begin_bias();
		const_iterator begin_bias() const;

		iterator	   end_bias();
		const_iterator end_bias() const;

	public:
			  Matrix& operator[](size_t index);
		const Matrix& operator[](size_t index) const;
   
	public:
			  Matrix& at_bias(size_t index);
		const Matrix& at_bias(size_t index) const;
 
	public:
			  Matrix& back();
		const Matrix& back() const;
	
	public:
			  Matrix& back_bias();
		const Matrix& back_bias() const;
	
	public:
		void push_back(const Matrix& );
		void push_back_bias(const Matrix& );
	
	public:
		size_t size() const;
		bool   empty() const; 
	  
	public:
		size_t blocks() const;
		size_t blockStep() const;

	public:
		void setBlockStep(size_t s);

	public:
		void setBias(const BlockSparseMatrix& bias);
		const BlockSparseMatrix& getBias() const;

	public:
		NeuronSet getInputNeuronsConnectedToTheseOutputs(
			const NeuronSet& outputs) const;
		Layer getSubgraphConnectedToTheseOutputs(
			const NeuronSet& outputs) const;

	private:
		BlockSparseMatrix m_sparseMatrix;
		BlockSparseMatrix m_bias;
		
	private:
		size_t m_blockStep;
		

};

}
}

