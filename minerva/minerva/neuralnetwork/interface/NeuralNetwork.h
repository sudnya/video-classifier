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

// Forward Declaration
namespace minerva { namespace neuralnetwork { class BackPropagation; } }

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
		typedef Layer::NeuronSet NeuronSet;
		
		typedef std::vector<Layer> LayerVector;

		typedef LayerVector::reverse_iterator	    reverse_iterator;
		typedef LayerVector::iterator	            iterator;
		typedef LayerVector::const_iterator         const_iterator;
		typedef LayerVector::const_reverse_iterator const_reverse_iterator;

	public:
		NeuralNetwork();

	public:
		void initializeRandomly(std::default_random_engine& engine, float epsilon = 6.0f);
		void initializeRandomly(float epsilon = 6.0f);
		
		void train(const Matrix& input, const Matrix& reference);
		void train(Matrix&& input, Matrix&& reference);
		void train(BlockSparseMatrix& input, BlockSparseMatrix& reference);		

		Matrix runInputs(const Matrix& m) const;
		BlockSparseMatrix runInputs(const BlockSparseMatrix& m) const;

	public:
		float computeAccuracy(const Matrix& input, const Matrix& reference) const;
		float computeAccuracy(const BlockSparseMatrix& input, const BlockSparseMatrix& reference) const;
		
	public:
		std::string getLabelForOutputNeuron(unsigned int idx) const;
		void setLabelForOutputNeuron(unsigned int idx, const std::string& label);

	public:
		void setLabelsForOutputNeurons(const NeuralNetwork& network);

	public:
		void mirror();
		void cutOffSecondHalf();

	public:
		size_t getInputCount()  const;
		size_t getOutputCount() const;

		size_t getOutputCountForInputCount(size_t inputCount) const;

		size_t getOutputNeurons() const;

		size_t getInputBlockingFactor()  const;
		size_t getOutputBlockingFactor() const;
	
	public:
		size_t totalNeurons()     const;
		size_t totalConnections() const;

	public:
		size_t getFloatingPointOperationCount() const;

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
		void addLayer(Layer&&);
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
		reverse_iterator       rbegin();
		const_reverse_iterator rbegin() const;

		reverse_iterator       rend();
		const_reverse_iterator rend() const;

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
		size_t size() const;
		bool   empty() const;

	public:
		void setUseSparseCostFunction(bool shouldUse);
		bool isUsingSparseCostFunction() const;

	public:
		BackPropagation* createBackPropagation() const;
	
	public:
		bool areConnectionsValid() const;

	public:
		NeuronSet getInputNeuronsConnectedToThisOutput(unsigned neuron) const;
		NeuralNetwork getSubgraphConnectedToThisOutput(unsigned neuron) const;

	public:
		std::string shapeString() const;

	private:
		LayerVector m_layers;
		NeuronToLabelMap m_labels;
	
	private:
		bool _useSparseCostFunction;

};

}//end neural network
}//end minerva

