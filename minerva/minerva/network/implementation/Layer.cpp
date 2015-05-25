/* Author: Sudnya Padalikar
 * Date  : 08/11/2013
 * The implementation of the layer class
 */

// Minerva Includes
#include <minerva/network/interface/Layer.h>

#include <minerva/network/interface/ActivationCostFunction.h>
#include <minerva/network/interface/ActivationFunction.h>
#include <minerva/network/interface/WeightCostFunction.h>

#include <minerva/network/interface/ActivationCostFunctionFactory.h>
#include <minerva/network/interface/ActivationFunctionFactory.h>
#include <minerva/network/interface/WeightCostFunctionFactory.h>

#include <minerva/matrix/interface/Matrix.h>
#include <minerva/matrix/interface/MatrixTransformations.h>

// Standard Library Includes
#include <sstream>

namespace minerva
{

namespace network
{

Layer::Layer()
: _activationFunction(ActivationFunctionFactory::create()),
  _activationCostFunction(ActivationCostFunctionFactory::create()),
  _weightCostFunction(WeightCostFunctionFactory::create())
{

}

Layer::~Layer()
{

}

Layer::Layer(const Layer& l)
: _activationFunction(l.getActivationFunction()->clone()),
  _activationCostFunction(l.getActivationCostFunction() == nullptr ? nullptr : l.getActivationCostFunction()->clone()),
  _weightCostFunction(l.getWeightCostFunction() == nullptr ? nullptr : l.getWeightCostFunction()->clone())
{

}

Layer& Layer::operator=(const Layer& l)
{
	if(&l == this)
	{
		return *this;
	}

	setActivationFunction(l.getActivationFunction()->clone());
	setActivationCostFunction(l.getActivationCostFunction() == nullptr ? nullptr : l.getActivationCostFunction()->clone());
	setWeightCostFunction(l.getWeightCostFunction() == nullptr ? nullptr : l.getWeightCostFunction()->clone());

	return *this;
}

static bool compareSize(matrix::Dimension inputSize, matrix::Dimension expectedSize)
{
    inputSize.pop_back();
    expectedSize.pop_back();

    return inputSize == expectedSize;
}

static bool compareCount(matrix::Dimension inputSize, size_t inputCount)
{
    inputSize.pop_back();

    return inputSize.product() == inputCount;
}

static matrix::Matrix reshapeExceptMinibatch(const matrix::Matrix& m, matrix::Dimension size, size_t minibatch)
{
    size.pop_back();
    size.push_back(minibatch);

    return reshape(m, size);
}

matrix::Matrix Layer::runForward(const Matrix& m) const
{
    if(compareSize(m.size(), getInputSize()))
    {
        return runForwardImplementation(m);
    }

    if(compareCount(m.size(), getInputCount()))
    {
        return runForwardImplementation(reshapeExceptMinibatch(m, getInputSize(), m.size().back()));
    }

    throw std::runtime_error("Input activation matrix size " + m.size().toString() + " is not compatible with layer, expecting " + getInputSize().toString());
}

matrix::Matrix Layer::runReverse(MatrixVector& gradients,
    const Matrix& inputActivations,
    const Matrix& outputActivations,
    const Matrix& deltas) const
{
    return runReverseImplementation(gradients,
        reshapeExceptMinibatch(inputActivations, getInputSize(), inputActivations.size().back()),
        outputActivations,
        reshapeExceptMinibatch(deltas, getOutputSize(), deltas.size().back()));
}

void Layer::setActivationFunction(ActivationFunction* f)
{
	_activationFunction.reset(f);
}

ActivationFunction* Layer::getActivationFunction()
{
	return _activationFunction.get();
}

const ActivationFunction* Layer::getActivationFunction() const
{
	return _activationFunction.get();
}

void Layer::setActivationCostFunction(ActivationCostFunction* f)
{
	_activationCostFunction.reset(f);
}

ActivationCostFunction* Layer::getActivationCostFunction()
{
	return _activationCostFunction.get();
}

const ActivationCostFunction* Layer::getActivationCostFunction() const
{
	return _activationCostFunction.get();
}

void Layer::setWeightCostFunction(WeightCostFunction* f)
{
	_weightCostFunction.reset(f);
}

WeightCostFunction* Layer::getWeightCostFunction()
{
	return _weightCostFunction.get();
}

const WeightCostFunction* Layer::getWeightCostFunction() const
{
	return _weightCostFunction.get();
}

std::string Layer::shapeString() const
{
	std::stringstream stream;

	stream << "(" << getTypeName() << " type, "
		<< getInputCount()
		<< " inputs, " << getOutputCount() << " outputs)";

	return stream.str();
}

}

}


