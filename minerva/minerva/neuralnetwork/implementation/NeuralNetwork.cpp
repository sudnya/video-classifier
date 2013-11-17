/* Author: Sudnya Padalikar
 * Date  : 08/09/2013
 * The implementation of the Neural Network class 
 */

#include <minerva/optimizer/interface/Solver.h>
#include <minerva/neuralnetwork/interface/NeuralNetwork.h>
#include <minerva/neuralnetwork/interface/BackPropData.h>
#include <minerva/util/interface/debug.h>

#include <cassert>
#include <vector>

typedef minerva::optimizer::Solver Solver;

namespace minerva
{
namespace neuralnetwork
{

void NeuralNetwork::initializeRandomly(float epsilon)
{
    util::log("NeuralNetwork") << "Initializing neural network randomly.\n";
    
    for (auto i = m_layers.begin(); i != m_layers.end(); ++i)
    {
        (*i).initializeRandomly(epsilon);
    }
}

void NeuralNetwork::train(const Matrix& inputMatrix, const Matrix& referenceOutput)
{
    //create a backpropagate-data class
    //given the neural network, inputs & reference outputs 
    util::log("NeuralNetwork") << "Running back propagation on input matrix (" << inputMatrix.rows() << ") rows, ("
       << inputMatrix.columns() << ") columns. Using reference output of (" << referenceOutput.rows() << ") rows, ("
       << referenceOutput.columns() << ") columns. \n";

    BackPropData data(this, convertInputToBlockSparse(inputMatrix), convertOutputToBlockSparse(referenceOutput));
    //should we worry about comparing costs here?
    //data.computeCost();
    //data.computeDerivative();

    Solver* solver = Solver::create(&data);
    
    try
    {
        solver->solve();
    }
    catch(...)
    {
        delete solver;

        throw;
    }

    delete solver;
}

NeuralNetwork::Matrix NeuralNetwork::runInputs(const Matrix& m) const
{
    //util::log("NeuralNetwork") << "Running forward propagation on matrix (" << m.rows()
    //        << " rows, " << m.columns() << " columns).\n";
    
    auto temp = convertInputToBlockSparse(m);

	return convertOutputToMatrix(runInputs(temp));
}

NeuralNetwork::BlockSparseMatrix NeuralNetwork::runInputs(const BlockSparseMatrix& m) const
{
	auto temp = m;

    for (auto i = m_layers.begin(); i != m_layers.end(); ++i)
    {
        temp = (*i).runInputs(temp);
    }

    return temp;
}

float NeuralNetwork::computeAccuracy(const Matrix& input, const Matrix& reference) const
{
	return computeAccuracy(convertInputToBlockSparse(input),
		convertOutputToBlockSparse(reference));
}

float NeuralNetwork::computeAccuracy(const BlockSparseMatrix& input,
	const BlockSparseMatrix& reference) const
{
	assert(input.rows() == reference.rows());
	assert(reference.columns() == getOutputCount());

	auto result = runInputs(input);

	float threshold = 0.5f;

	auto resultActivations    = result.greaterThanOrEqual(threshold);
	auto referenceActivations = result.greaterThanOrEqual(threshold);

	auto matchingActivations = resultActivations.equals(referenceActivations);

	float matches = matchingActivations.reduceSum();

	return matches / result.size();
}

std::string NeuralNetwork::getLabelForOutputNeuron(unsigned int i) const
{
    auto label = m_labels.find(i);
    
    if(label == m_labels.end())
    {
    	std::stringstream stream;
    	
    	stream << "output-neuron-" << i;
    	
    	return stream.str();
    }
    
    return label->second;
}

void NeuralNetwork::setLabelForOutputNeuron(unsigned int idx, const std::string& label)
{
	assert(idx < getOutputCount());

    m_labels[idx] = label;
}

void NeuralNetwork::addLayer(const Layer& L)
{
    m_layers.push_back(L);
}

unsigned NeuralNetwork::getTotalLayerSize() const
{
    return m_layers.size();
}

const NeuralNetwork::LayerVector* NeuralNetwork::getLayers() const
{
    return &m_layers;
}

NeuralNetwork::LayerVector* NeuralNetwork::getLayers()
{
    return &m_layers;
}

void NeuralNetwork::resize(size_t layers)
{
	m_layers.resize(layers);
}

void NeuralNetwork::mirror()
{
	addLayer(Layer(1, getOutputCount(), getInputCount()));
	
    // should be pseudo inverse
    back().initializeRandomly();   
}

void NeuralNetwork::cutOffSecondHalf()
{
	assert(size() > 1);

	resize(size() - 1);
}

unsigned NeuralNetwork::getInputCount() const
{
    if(empty())
        return 0;

    return front().getInputCount();
}

unsigned NeuralNetwork::getOutputCount() const
{
    if(empty())
        return 0;

    return back().getOutputCount();
}

size_t NeuralNetwork::totalWeights() const
{
	size_t weights = 0;
	
	for(auto& layer : *this)
	{
		weights += layer.totalWeights();
	}
	
	return weights;
}

NeuralNetwork::Matrix NeuralNetwork::getFlattenedWeights() const
{
	Matrix weights;
	
	for(auto& layer : *this)
	{
		weights = weights.appendColumns(layer.getFlattenedWeights());
	}
	
	return weights;
}

void NeuralNetwork::setFlattenedWeights(const Matrix& m)
{
	size_t weights = 0;

	for(auto& layer : *this)
	{
		auto sliced = m.slice(0, weights, 1, layer.totalWeights());
		
		layer.setFlattenedWeights(sliced);
		
		weights += layer.totalWeights();
	}
}

NeuralNetwork::Matrix NeuralNetwork::convertOutputToMatrix(const BlockSparseMatrix& blockedMatrix) const
{
	Matrix result;
	
	for(auto& matrix : blockedMatrix)
	{
		result = result.appendRows(matrix);
	}
	
	return result;
}

NeuralNetwork::BlockSparseMatrix NeuralNetwork::convertInputToBlockSparse(const Matrix& m) const
{
	assert(m.columns() == front().getInputCount());
	
	BlockSparseMatrix result;
	size_t column = 0;

	for(auto& block : front())
	{
		size_t blockInputsExceptBias = block.rows();
		result.push_back(m.slice(0, column, m.rows(), column + blockInputsExceptBias));
		column += block.rows();
	}
	
	return result;
}

NeuralNetwork::BlockSparseMatrix NeuralNetwork::convertOutputToBlockSparse(const Matrix& m) const
{
	assert(m.columns() == back().getOutputCount());
	
	BlockSparseMatrix result;
	size_t column = 0;

	for(auto& block : back())
	{
		result.push_back(m.slice(0, column, m.rows(), column + block.columns()));
		column += block.columns();
	}
	
	return result;
}

NeuralNetwork::iterator NeuralNetwork::begin()
{
	return m_layers.begin();
}

NeuralNetwork::const_iterator NeuralNetwork::begin() const
{
	return m_layers.begin();
}

NeuralNetwork::iterator NeuralNetwork::end()
{
	return m_layers.end();
}

NeuralNetwork::const_iterator NeuralNetwork::end() const
{
	return m_layers.end();
}

NeuralNetwork::Layer& NeuralNetwork::operator[](size_t index)
{
	return m_layers[index];
}

const NeuralNetwork::Layer& NeuralNetwork::operator[](size_t index) const
{
	return m_layers[index];
}

Layer& NeuralNetwork::back()
{
	return m_layers.back();
}

const Layer& NeuralNetwork::back() const
{
	return m_layers.back();
}

Layer& NeuralNetwork::front()
{
	return m_layers.front();
}

const Layer& NeuralNetwork::front() const
{
	return m_layers.front();
}

unsigned int NeuralNetwork::size() const
{
	return m_layers.size();
}

bool NeuralNetwork::empty() const
{
	return m_layers.empty();
}

}

}

