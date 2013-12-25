/* Author: Sudnya Padalikar
 * Date  : 08/09/2013
 * The interface of the Neural Network class 
 */

#pragma once

#include <minerva/matrix/interface/Matrix.h>
#include <minerva/neuralnetwork/interface/Layer.h>

#include <string>
#include <vector>
#include <map>
#include <random>

namespace minerva
{
namespace neuralnetwork
{

class NeuralNetwork
{
	public:
		typedef minerva::matrix::Matrix Matrix;
		typedef minerva::matrix::BlockSparseMatrix BlockSparseMatrix;
		typedef minerva::neuralnetwork::Layer Layer;

		typedef std::map<unsigned, std::string> NeuronToLabelMap;
		
		typedef std::vector<Layer> LayerVector;

		typedef LayerVector::iterator	   iterator;
		typedef LayerVector::const_iterator const_iterator;

	public:
		NeuralNetwork()
		{
		}
	
		void initializeRandomly(std::default_random_engine& engine, float epsilon = 0.3f);
		void initializeRandomly(float epsilon = 0.3f);
		
		void train(const Matrix& input, const Matrix& reference);
		void train(const BlockSparseMatrix& input, const BlockSparseMatrix& reference);		

		Matrix runInputs(const Matrix& m) const;
		BlockSparseMatrix runInputs(const BlockSparseMatrix& m) const;

	public:
		float computeAccuracy(const Matrix& input, const Matrix& reference) const;
		float computeAccuracy(const BlockSparseMatrix& input, const BlockSparseMatrix& reference) const;
		
	public:
		std::string getLabelForOutputNeuron(unsigned int idx) const;
		void setLabelForOutputNeuron(unsigned int idx, const std::string& label);

	public:
		void mirror();
		void cutOffSecondHalf();

	public:
		unsigned getInputCount()  const;
		unsigned getOutputCount() const;

		unsigned getBlockingFactor() const;

	public:
		size_t totalWeights()     const;
		size_t totalActivations() const;
	
	public:
		Matrix getFlattenedWeights() const;
		void setFlattenedWeights(const Matrix& m);

	public:
		BlockSparseMatrix convertToBlockSparseForLayerInput(const Layer& layer, const Matrix& m) const;
		BlockSparseMatrix convertToBlockSparseForLayerOutput(const Layer& layer, const Matrix& m) const;
		BlockSparseMatrix convertOutputToBlockSparse(const Matrix& m) const;
		void formatInputForLayer(const Layer& layer, BlockSparseMatrix& m) const;
		void formatOutputForLayer(const Layer& layer, BlockSparseMatrix& m) const;

	public:
		void addLayer(const Layer&);
		unsigned getTotalLayerSize() const;
		
		      LayerVector* getLayers();
		const LayerVector* getLayers() const;

	public:
		void resize(size_t layers);

		void clear();

	public:
		iterator       begin();
		const_iterator begin() const;

		iterator       end();
		const_iterator end() const;

	public:
		      Layer& operator[](size_t index);
		const Layer& operator[](size_t index) const;

	public:
		      Layer& back();
		const Layer& back() const;
	
	public:
		      Layer& front();
		const Layer& front() const;
	
	public:
		unsigned int size() const;
		bool         empty() const;
	
	private:
		LayerVector m_layers;
		NeuronToLabelMap m_labels;

};

}//end neural network
}//end minerva

