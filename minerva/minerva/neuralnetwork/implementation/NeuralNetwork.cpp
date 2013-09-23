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

void NeuralNetwork::backPropagate(const Matrix& inputMatrix, const Matrix& referenceOutput)
{
    //create a backpropagate-data class
    //given the neural network, inputs & reference outputs 
    util::log("NeuralNetwork") << "Running back propagation on input matrix (" << inputMatrix.rows() << ") rows, ("
       << inputMatrix.columns() << ") columns. Using reference output of (" << referenceOutput.rows() << ") rows, ("
       << referenceOutput.columns() << ") columns. \n";

    BackPropData data(this, inputMatrix, referenceOutput);
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
    
    Matrix temp = m;

    for (auto i = m_layers.begin(); i != m_layers.end(); ++i)
    {
        temp = (*i).runInputs(temp);
    }

    return temp;
}

float NeuralNetwork::computeAccuracy(const Matrix& input, const Matrix& reference) const
{
	assert(input.rows() == reference.rows());
	assert(reference.columns() == getOutputCount());

	Matrix result = runInputs(input);

	float matches = 0.0f;

	for(Matrix::const_iterator referenceValue = reference.begin(), resultValue = result.begin();
		resultValue != result.end(); ++referenceValue, ++resultValue)
	{
		bool referenceActivation = *referenceValue >= 0.5f;
		bool resultActivation    = *resultValue    >= 0.5f;

		if(referenceActivation == resultActivation)
		{
			matches += 1.0f;
		}
	}

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

