/* Author: Sudnya Padalikar
 * Date  : 08/11/2013
 * The implementation of the layer class
 */

// Lucious Includes
#include <lucious/network/interface/Layer.h>

#include <lucious/network/interface/ActivationCostFunction.h>
#include <lucious/network/interface/ActivationFunction.h>
#include <lucious/network/interface/WeightCostFunction.h>

#include <lucious/network/interface/ActivationCostFunctionFactory.h>
#include <lucious/network/interface/ActivationFunctionFactory.h>
#include <lucious/network/interface/WeightCostFunctionFactory.h>

#include <lucious/matrix/interface/Matrix.h>
#include <lucious/matrix/interface/MatrixVector.h>
#include <lucious/matrix/interface/MatrixTransformations.h>

#include <lucious/util/interface/PropertyTree.h>

// Standard Library Includes
#include <sstream>

namespace lucious
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
    size_t minimumSize = expectedSize.size() - 2;

    expectedSize.pop_back(2);

    while(inputSize.size() > minimumSize)
    {
        inputSize.pop_back();
    }

    return inputSize == expectedSize;
}

static bool compareCount(matrix::Dimension inputSize, matrix::Dimension expectedSize, size_t inputCount)
{
    size_t product = 1;

    for(size_t dimension = 0; dimension < inputSize.size(); ++dimension)
    {
        product *= inputSize[dimension];

        if(product == inputCount)
        {
            return true;
        }
    }

    return false;
}

static void removeTimeAndBatch(matrix::Dimension& dimension)
{
    // remove time and batch dimensions
    dimension.pop_back(2);

}

static matrix::Matrix reshapeActivations(const matrix::Matrix& m, matrix::Dimension expectedSize)
{
    auto newShape = expectedSize;

    removeTimeAndBatch(newShape);

    // add back batch
    newShape.push_back(m.size()[m.size().size() - 2]);

    // add back time
    newShape.push_back(m.size()[m.size().size() - 1]);

    return reshape(m, newShape);
}

void Layer::runForward(MatrixVector& activations) const
{
    if(compareSize(activations.back().size(), getInputSize()))
    {
        return runForwardImplementation(activations);
    }

    if(compareCount(activations.back().size(), getInputSize(), getInputCount()))
    {
        activations.back() = reshapeActivations(activations.back(), getInputSize());

        return runForwardImplementation(activations);
    }

    throw std::runtime_error("Input activation matrix size " + activations.back().size().toString() +
        " is not compatible with layer, expecting " + getInputSize().toString());
}

matrix::Matrix Layer::runReverse(MatrixVector& gradients,
    MatrixVector& activations,
    const Matrix& deltas) const
{
    if(compareCount(activations.back().size(), getOutputSize(), getOutputCount()))
    {
        activations.back() = reshapeActivations(activations.back(), getOutputSize());
    }

    return runReverseImplementation(gradients,
        activations,
        reshapeActivations(deltas, getOutputSize()));
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

void Layer::saveLayer(util::OutputTarArchive& archive, util::PropertyTree& properties) const
{
    properties["activation-function"]      = getActivationFunction()->typeName();
    properties["activation-cost-function"] = getActivationCostFunction()->typeName();
    properties["weight-cost-function"]     = getWeightCostFunction()->typeName();
}

void Layer::loadLayer(util::InputTarArchive& archive, const util::PropertyTree& properties)
{
    setActivationFunction(ActivationFunctionFactory::create(properties["activation-function"]));
    setActivationCostFunction(ActivationCostFunctionFactory::create(properties["activation-cost-function"]));
    setWeightCostFunction(WeightCostFunctionFactory::create(properties["weight-cost-function"]));
}

}

}


